"""
Generate ``sdo/aia/data/aia_chianti11_temperature_response.ecsv``.

The temperature response of the AIA EUV channels computed from version 11 of
CHIANTI, with the lines from :mod:`utu.spectrum` and the continua from
:mod:`fiasco`. Everything but the atomic data is that of the SolarSoft
version 10 response, which ``tools/aia_temperature_response.py`` generates, so
that the two tables differ by the version of CHIANTI alone:

- the same effective areas, from ``aia_V9_all_fullinst.genx``, with the same
  crosstalk and conversion to DN;
- the same plasma, as ``aia_V9_fullemiss.genx`` records it: the coronal
  abundances of Feldman (1992), a constant pressure of 1e15 K cm^-3, and
  proton rates, with the ionization equilibrium of the version of CHIANTI;
- the same temperatures, log T from 4 to 9 in steps of 0.05;
- the same emission: every bound-bound line of every element at least as
  abundant as zinc, and the free-free, free-bound and two-photon continua;
- the same empirical boost of the He II 303.786 angstrom line by a factor of
  20, which ``aia_bp_make_emiss`` applies to stand in for the optical depth of
  the line.

There is nothing like ``/chiantifix``, which corrects for lines missing from
the version of CHIANTI behind version 10, and so is not part of this table.

Each ion is computed in a process of its own, and its contribution to every
channel is saved in ``DIRECTORY/chianti_VERSION_DIGEST`` as soon as it is
done, so an interrupted run resumes where it stopped. ``DIGEST`` changes with
this script, the one it imports, the file of effective areas, and the
versions of :mod:`fiasco` and :mod:`utu`, so a run after any of them changes
starts over instead of reusing what they computed before. The whole database
takes several hours of CPU time, almost all of it solving for level
populations.

Besides this package, it needs :mod:`fiasco` and :mod:`utu`, and the HDF5
database of CHIANTI which :mod:`fiasco` builds. The database is the one
:mod:`fiasco` reads by default, or another given with ``--database``.

Given the database of CHIANTI 9.0.1, the version behind the version 10
response, this computes the version 10 response from scratch, which checks
everything here but the version of CHIANTI. Done on 2026-10-09, the result
agreed with the line list and continuum of ``aia_V9_fullemiss.genx`` folded
the same way to 0.2% rms in every channel, and ion by ion to 0.6%, and with
the version 10 table to between 0.1% and 3.1% rms, depending on the channel.
What is left against the table comes from the binning of
``aia_bp_make_emiss``, which moves each line up to 0.05 angstroms to the
nearest step of 0.1 angstroms, worth 2% for the hot lines of 94, 131 and 335
angstroms on the steep edges of their effective areas, and drops the lines
which fall exactly halfway between two steps, worth 7% in 193 angstroms at
log T 5.4.

Usage::

    python tools/aia_temperature_response_chianti.py DIRECTORY [--workers N]
        [--memory GB] [--database HDF5] [--output ECSV]
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import os
import pathlib
import sys
import time
import warnings
from typing import Any, cast

import numpy as np
import astropy.constants
import astropy.table
import astropy.units as u
import fiasco
import fiasco.util.exceptions
import named_arrays as na
import utu.spectrum
from sunpy.io.special import read_genx

import aia_temperature_response as ssw

logte = np.round(np.arange(4, 9 + 1e-9, 0.05), 2)
"""The log temperatures of the response, those of the version 10 table."""

pressure = 1e15 * u.K / u.cm**3
"""The constant pressure, :math:`n_e T`, of the plasma."""

abundance = "sun_coronal_1992_feldman_ext"
"""The abundances, by their name in :mod:`fiasco`."""

abundance_min = 1e-7
"""
The abundance, relative to hydrogen, below which an element is left out.
Zinc, the least abundant element in the version 10 emissivity, is just above
it.
"""

ionization_fraction = "chianti"
"""The ionization equilibrium, by its name in :mod:`fiasco`."""

boost = {("He 2", "1s 2S1/2", "2p 2P1/2"): 20}
"""
The factor each line is multiplied by, keyed by the ion and the labels
:mod:`fiasco` gives the lower and upper levels of the line, made from the
energy levels of CHIANTI rather than the labels of its lines, which are
wrong for these levels of He II in version 11. ``aia_bp_make_emiss`` boosts
the :math:`1s\\,^2S_{1/2} - 2p\\,^2P_{1/2}` line of He II at 303.786
angstroms, one of the two lines of the doublet, by a factor of 20.
"""

workers_max_windows = 61
"""
The most processes :class:`concurrent.futures.ProcessPoolExecutor` can have
on Windows.
"""

components = ("lines", "free_free", "free_bound", "two_photon")
"""The kinds of emission each ion contributes."""

output = (
    pathlib.Path(__file__).parents[1]
    / "sdo/aia/data/aia_chianti11_temperature_response.ecsv"
)

_hc = (astropy.constants.h * astropy.constants.c).to_value(u.erg * u.AA)
"""Planck's constant times the speed of light, in erg angstrom."""

_axis = "temperature"
"""The name of the axis along the temperatures, for :mod:`utu`."""

_state: dict[str, Any] = {}
"""What every process needs, set once when it starts by :func:`_initialize`."""


def _initialize(
    proton_electron_ratio: np.ndarray,
    wavelength: np.ndarray,
    area: np.ndarray,
    database: pathlib.Path,
) -> None:
    """Store what every ion needs in the process computing it."""
    warnings.simplefilter("ignore")
    fiasco.log.setLevel("ERROR")
    _state.update(
        proton_electron_ratio=proton_electron_ratio,
        wavelength=wavelength,
        area=area,
        database=database,
    )

    # The continua vary slowly, so they are summed over the wavelengths the
    # effective area is tabulated at, as ``aia_bp_make_tresp`` sums the
    # emissivity, with these weights.
    step = (wavelength[-1] - wavelength[0]) / (wavelength.size - 1)
    weight = _area(wavelength) * (wavelength / _hc * step)[:, np.newaxis]
    _state.update(weight=weight)


def _area(wavelength: np.ndarray) -> np.ndarray:
    """
    The effective area of every channel at the given wavelengths, zero outside
    the wavelengths it is tabulated at, as ``aia_bp_make_tresp`` takes it.

    Has the shape of `wavelength` plus an axis along the channels, in DN cm^2
    sr pix^-1 per photon, including the plate scale and the :math:`1/4\\pi` of
    an isotropic source.
    """
    w = _state["wavelength"]
    area = _state["area"]
    result = [
        np.interp(wavelength, w, area[:, j], left=0, right=0)
        for j in range(area.shape[1])
    ]
    return np.maximum(np.stack(result, axis=-1), 0)


def _lines(ion: fiasco.Ion, density: u.Quantity) -> tuple[np.ndarray, int]:
    """
    The contribution of the lines of an ion to every channel, with the shape
    (temperature, channel), and the number of lines it was summed over.
    """
    w = _state["wavelength"]
    g = utu.spectrum.contribution_function(
        ion=ion,
        density=na.ScalarArray(density, axes=_axis),
        axis_temperature=_axis,
        proton_electron_ratio=_state["proton_electron_ratio"],
    )
    wavelength = g.inputs.ndarray.to_value(u.AA)
    g = g.outputs.ndarray_aligned((_axis, "line"))
    g = g.to_value(u.erg * u.cm**3 / u.s)

    factor = np.ones(wavelength.shape)
    for (name, lower, upper), f in boost.items():
        if name == ion.ion_name:
            # the lines of the result are the bound-bound transitions
            transitions = ion.transitions
            bound = transitions.is_bound_bound
            where = (transitions.lower_level[bound] == _level(ion, lower)) & (
                transitions.upper_level[bound] == _level(ion, upper)
            )
            if where.sum() != 1:
                raise ValueError(f"{name} has {where.sum()} lines {lower}-{upper}")
            factor[where] = f

    where = (wavelength >= w.min()) & (wavelength <= w.max())
    photons = g[:, where] * factor[where] * wavelength[where] / _hc
    return photons @ _area(wavelength[where]), int(where.sum())


def _level(ion: fiasco.Ion, label: str) -> int:
    """The number of the one level of an ion with a label like ``"2p 2P1/2"``."""
    levels = ion.levels
    where = levels.label == label
    if where.sum() != 1:
        raise ValueError(f"{ion.ion_name} has {where.sum()} levels {label!r}")
    return int(levels.level[where][0])


def _two_photon(
    ion: fiasco.Ion,
    wavelength: u.Quantity,
    density: u.Quantity,
) -> u.Quantity:
    """
    The two-photon continuum of an ion at one density per temperature, with
    the shape (temperature, wavelength).

    With ``couple_density_to_temperature=True``, :mod:`fiasco` 0.8.2 takes the
    population of the upper level with the density axis kept, of shape
    (temperature, 1, 1), and divides it by a density of shape (temperature,
    1), which broadcasts to (temperature, temperature, 1) and fails. Handing
    it the populations without the density axis gives the shape the rest of
    the method expects. Solving one temperature at a time instead costs fifty
    times as much, since every solve rebuilds the rates.

    The populations stand in for the call ``two_photon`` makes in
    :mod:`fiasco` 0.8.2, and any other call is an error, so that a change in
    :mod:`fiasco` stops this rather than giving the wrong populations.
    """
    # as fiasco computes the two-photon continuum, without protons
    expected = dict(couple_density_to_temperature=True, include_protons=False)
    populations = ion.level_populations(density, **expected)

    def level_populations(
        electron_density: u.Quantity,
        **kwargs: object,
    ) -> u.Quantity:
        """The populations solved above, without the density axis."""
        if kwargs != expected or not np.array_equal(electron_density, density):
            raise RuntimeError(
                "fiasco's two_photon no longer asks for the level populations "
                "this workaround solves for: "
                f"{kwargs=}, {electron_density=}"
            )
        return populations[:, 0]

    ion.level_populations = level_populations
    try:
        result = ion.two_photon(wavelength, density, couple_density_to_temperature=True)
    finally:
        del ion.level_populations
    return result[:, 0]


def ion_response(name: str, lines: bool) -> dict[str, np.ndarray]:
    """
    The contribution of one ion to the response of every channel.

    Each of :data:`components` has the shape (temperature, channel), in DN
    cm^5 s^-1 pix^-1 per unit emission measure of :math:`n_e n_H`. The lines
    are only computed if `lines` is true, for an ion with lines in range.
    """
    start = time.perf_counter()
    temperature = 10**logte * u.K
    density = pressure / temperature
    ion = fiasco.Ion(
        name,
        temperature,
        abundance=abundance,
        ionization_fraction=ionization_fraction,
        hdf5_dbase_root=_state["database"],
    )

    shape = (logte.size, _state["area"].shape[1])
    result = {component: np.zeros(shape) for component in components}
    missing = []

    n_lines = 0
    if lines:
        try:
            result["lines"], n_lines = _lines(ion, density)
        except fiasco.util.exceptions.MissingDatasetException as e:
            missing.append(f"lines: {e}")

    weight = _state["weight"]
    unit = u.erg * u.cm**3 / u.s / u.AA
    wavelength = _state["wavelength"] * u.AA
    if ion.ionization_stage > 1:
        try:
            # Per steradian in fiasco 0.8.2, as in IDL, unlike its lines and
            # other continua, which `_check_free_free` makes sure of.
            ff = 4 * np.pi * ion.free_free(wavelength)
            ff = ff * ion.abundance * ion.ionization_fraction[:, np.newaxis]
            result["free_free"] = ff.to_value(unit) @ weight
        except fiasco.util.exceptions.MissingDatasetException as e:
            missing.append(f"free-free: {e}")
    try:
        fb = ion.free_bound(wavelength)
        fraction = ion.next_ion().ionization_fraction
        fb = fb * ion.abundance * fraction[:, np.newaxis]
        result["free_bound"] = fb.to_value(unit) @ weight
    except fiasco.util.exceptions.MissingDatasetException as e:
        missing.append(f"free-bound: {e}")
    if ion.hydrogenic or ion.helium_like:
        try:
            tp = _two_photon(ion, wavelength, density)
            tp = tp * ion.abundance * ion.ionization_fraction[:, np.newaxis]
            result["two_photon"] = tp.to_value(unit) @ weight
        except fiasco.util.exceptions.MissingDatasetException as e:
            missing.append(f"two-photon: {e}")

    result["n_lines"] = np.array(n_lines)
    result["missing"] = np.array(missing, dtype=str)
    result["seconds"] = np.array(time.perf_counter() - start)
    return result


def _ions(database: pathlib.Path) -> list[tuple[str, bool, float]]:
    """
    Every ion of every element at least as abundant as :data:`abundance_min`,
    whether it has lines within the effective area, and an estimate of the
    memory computing it takes in gigabytes, the largest first, so that the
    slowest ions do not start last.
    """
    # utu.spectrum.ions lists the ions of the default database of fiasco,
    # whatever the database it is given, so it would miss the lines of an
    # ion only `database` has.
    extra = set(fiasco.list_ions(hdf5_dbase_root=database)) - set(fiasco.list_ions())
    if extra:
        raise ValueError(
            f"{database} has ions the default database of fiasco does not, "
            f"which utu.spectrum.ions would leave out: {sorted(extra)}"
        )

    with_lines = set(
        utu.spectrum.ions(
            wavelength_min=25 * u.AA,
            wavelength_max=900 * u.AA,
            abundance_min=abundance_min,
            abundance=abundance,
            hdf5_dbase_root=database,
        )
    )
    result = []
    for name in fiasco.list_ions(hdf5_dbase_root=database):
        name = str(name)
        ion = fiasco.Ion(name, 1 * u.MK, abundance=abundance, hdf5_dbase_root=database)
        try:
            if ion.abundance < abundance_min:
                continue
        except fiasco.util.exceptions.MissingDatasetException:
            continue
        gigabytes = 0.0
        if name in with_lines:
            gigabytes = _gigabytes(ion)
        result.append((gigabytes, name))
    result.sort(reverse=True)
    return [(name, name in with_lines, gigabytes) for gigabytes, name in result]


def _gigabytes(ion: fiasco.Ion) -> float:
    """
    An estimate of the memory solving for the level populations of an ion
    takes, in gigabytes.

    fiasco builds the rate matrices of every temperature at once, about
    eight arrays of (temperature, level, level) at their largest. An ion with
    autoionization or level-resolved recombination data is solved together
    with the next ion, the two-ion model, so its matrices span the levels of
    both. Measured, O VI (923 + 577 levels) peaks at 14.6 GB.
    """
    n = ion.n_levels
    if ion._has_dataset("auto") or ion._has_dataset("rrlvl"):
        try:
            n = n + ion.next_ion().n_levels
        except fiasco.util.exceptions.MissingDatasetException:
            pass
    return 8 * logte.size * n**2 * 8 / 1e9


def main(
    directory: pathlib.Path,
    workers: int,
    memory: float,
    database: pathlib.Path,
    output: pathlib.Path,
) -> None:
    warnings.simplefilter("ignore")
    fiasco.log.setLevel("ERROR")

    directory.mkdir(parents=True, exist_ok=True)
    path = ssw.download(directory, ssw.files["area"])
    area_file = read_genx(path)

    wavelength, _, _ = ssw.effective_area(area_file, ssw.channels[0])
    area = []
    for channel in ssw.channels:
        wave, effarea, platescale = ssw.effective_area(area_file, channel)
        if not np.array_equal(wave, wavelength):
            raise ValueError(
                f"the effective area of {channel} angstroms is tabulated at "
                f"other wavelengths than that of {ssw.channels[0]} angstroms"
            )
        # in double precision, since the file stores single
        effarea = effarea.astype(float)
        area.append(effarea * platescale / (4 * np.pi))
    area = np.stack(area, axis=-1)

    temperature = 10**logte * u.K
    ratio = fiasco.proton_electron_ratio(
        temperature,
        abundance=abundance,
        ionization_fraction=ionization_fraction,
        hdf5_dbase_root=database,
    )
    version = _chianti_version(database)
    _check_free_free(database)

    # One thread per process: the parallelism is over ions, and a BLAS that
    # starts a thread per core in every process slows the run to a crawl.
    # The processes inherit the environment when they start, before they
    # import numpy.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    done = directory / f"chianti_{version}_{_fingerprint(path)}"
    done.mkdir(exist_ok=True)
    tasks = _ions(database)
    results: dict[str, dict[str, np.ndarray]] = {}
    todo = []
    for name, lines, gigabytes in tasks:
        file = done / f"{name.replace(' ', '_')}.npz"
        if file.exists():
            results[name] = dict(np.load(file))
        else:
            todo.append((name, lines, gigabytes))
    print(f"{len(tasks)} ions, {len(todo)} to compute", flush=True)

    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=(
            ratio.to_value(u.dimensionless_unscaled),
            wavelength,
            area,
            database,
        ),
    ) as pool:
        # Submitted as the memory they need frees up, rather than all at
        # once, since the largest ions together need far more than a
        # computer has.
        running: dict[concurrent.futures.Future, tuple[str, float]] = {}
        k = 0
        while todo or running:
            used = sum(gigabytes for _, gigabytes in running.values())
            for task in list(todo):
                name, lines, gigabytes = task
                if len(running) >= workers:
                    break
                if running and used + gigabytes > memory:
                    continue
                running[pool.submit(ion_response, name, lines)] = (name, gigabytes)
                used += gigabytes
                todo.remove(task)
            finished, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                name, _ = running.pop(future)
                result = future.result()
                # written in full before it takes its name, so that a run
                # stopped while writing does not leave a file that looks done
                file = done / f"{name.replace(' ', '_')}.npz"
                partial = file.with_name(f"{file.name}.part")
                with open(partial, "wb") as f:
                    np.savez(f, **cast(dict[str, Any], result))
                partial.replace(file)
                results[name] = result
                k += 1
                elapsed = time.perf_counter() - start
                print(
                    f"{k}/{k + len(todo) + len(running)} {name}: "
                    f"{float(result['seconds']):.0f} s, "
                    f"{int(result['n_lines'])} lines, {elapsed:.0f} s elapsed",
                    flush=True,
                )

    # summed in the order of the tasks rather than the order they finished,
    # so that every run writes the same table
    results = {name: results[name] for name, _, _ in tasks}
    total = np.zeros((logte.size, len(ssw.channels)))
    for result in results.values():
        for component in components:
            total = total + result[component]
    missing = {
        name: list(r["missing"]) for name, r in results.items() if r["missing"].size
    }

    unit = u.DN * u.cm**5 / u.s / u.pix
    table = astropy.table.QTable()
    table["temperature"] = temperature
    for j, channel in enumerate(ssw.channels):
        table[f"response_{channel}"] = total[:, j] * unit

    table.meta = dict(
        channels=ssw.channels,
        description=(
            "Temperature response of the AIA EUV channels (thin focal-plane "
            f"filters, with crosstalk) from CHIANTI {version}, its lines, "
            "continua, and ionization equilibrium, with the effective areas, "
            "abundances, pressure, and He II boost of SolarSoft version 10, "
            "without the time-dependent corrections. response_N is the "
            "response of channel N. The emission measure is that of n_e n_H."
        ),
        files=[ssw.files["area"]],
        file_dates=dict(area=str(area_file["HEADER"].get("CREATION", ""))),
        chianti=dict(
            version=version,
            fiasco=fiasco.__version__,
            utu=_version("utu"),
            abundance=abundance,
            abundance_min=abundance_min,
            ionization_fraction=ionization_fraction,
            pressure=str(pressure),
            protons=True,
            boost={f"{k[0]} {k[1]} - {k[2]}": v for k, v in boost.items()},
            ions=len(results),
            lines=sum(int(r["n_lines"]) for r in results.values()),
        ),
        generated=datetime.date.today().isoformat(),
        generator="tools/aia_temperature_response_chianti.py",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write(output, format="ascii.ecsv", overwrite=True)
    print(f"wrote {output}")
    print(f"{len(missing)} ions are missing a component:")
    for name, why in missing.items():
        print(f"  {name}: {'; '.join(why)}")


def _check_free_free(database: pathlib.Path) -> None:
    """
    Check that the free-free continuum of :mod:`fiasco` is per steradian, as
    the factor of :math:`4\\pi` in :func:`ion_response` assumes.

    Its integral over wavelength is compared with the radiative loss
    :mod:`fiasco` computes from the integrated Gaunt factor, which is
    :math:`4\\pi` times as much, give or take the few percent between the two
    Gaunt factors.
    """
    ion = fiasco.Ion("H 2", [1e6, 1e7, 1e8] * u.K, hdf5_dbase_root=database)
    wavelength = np.geomspace(1e-3, 1e7, 4001) * u.AA
    integral = np.trapezoid(ion.free_free(wavelength), wavelength, axis=-1)
    ratio = ion.free_free_radiative_loss() / integral
    ratio = ratio.to_value(u.dimensionless_unscaled)
    if not np.allclose(ratio, 4 * np.pi, rtol=0.1):
        raise RuntimeError(
            "the free-free continuum of fiasco is no longer per steradian, "
            f"since its radiative loss is {ratio} times its integral over "
            "wavelength instead of 4 pi, so ion_response must not multiply "
            "it by 4 pi"
        )


def _fingerprint(area: pathlib.Path) -> str:
    """
    A digest of what the result of an ion depends on besides the database:
    the code of this script and of the one it imports, the file of effective
    areas, and the versions of :mod:`fiasco` and :mod:`utu`.
    """
    digest = hashlib.sha256()
    for path in (pathlib.Path(__file__), pathlib.Path(ssw.__file__), area):
        digest.update(path.read_bytes())
    for package in ("fiasco", "utu"):
        digest.update(_version(package).encode())
    return digest.hexdigest()[:12]


def _workers() -> int:
    """The number of processes to use by default, one per CPU."""
    result = os.cpu_count() or 1
    if sys.platform == "win32":
        result = min(result, workers_max_windows)
    return result


def _chianti_version(database: pathlib.Path) -> str:
    """The version of CHIANTI a database of :mod:`fiasco` was built from."""
    import h5py

    with h5py.File(database, "r") as f:
        return str(f["fe/fe_9/wgfa"].attrs["chianti_version"])


def _version(package: str) -> str:
    """The installed version of a package."""
    import importlib.metadata

    return importlib.metadata.version(package)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=_workers())
    parser.add_argument(
        "--memory",
        type=float,
        default=64,
        help="the gigabytes the ions computed at once may take together",
    )
    parser.add_argument(
        "--database",
        type=pathlib.Path,
        default=fiasco.defaults["hdf5_dbase_root"],
        help="the HDF5 database of fiasco to read, by default the one it reads",
    )
    parser.add_argument("--output", type=pathlib.Path, default=output)
    args = parser.parse_args()
    main(args.directory, args.workers, args.memory, args.database, args.output)
