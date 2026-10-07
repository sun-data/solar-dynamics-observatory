"""
Generate ``sdo/aia/data/aia_V10_temperature_response.ecsv``.

This is ``aia_get_response(/temperature, /dn)`` from the SolarSoft ``sdo/aia``
package, version 10, ported to Python, together with the empirical correction
``aia_get_response`` adds with ``/chiantifix``. It reads the same files the
IDL reads, downloading them from SolarSoft into ``directory`` if they are not
already there:

- ``aia_V9_all_fullinst.genx``, the effective area of each channel and its
  components (version 10 uses the version 9 file, itself a copy of version 8);
- ``aia_V9_fullemiss.genx``, the CHIANTI emissivity (790 MB, most of it the
  line list, which is not used here);
- ``aia_V9_chiantifix.genx``, the empirical correction.

The time-dependent corrections, the degradation and the EVE normalization,
are not in the table, since they are scalars per channel which
:func:`sdo.aia.temperature_response` applies from the correction table at the
time of an observation.

Each step names the IDL routine it ports. To check the result against IDL,
pass the ``.dat`` files written by ``aia_get_response`` (as in
``make_aiaresp_forpy.pro`` of ``demreg``) with ``--check``.

Usage::

    python tools/aia_temperature_response.py DIRECTORY [--check DIRECTORY_IDL]
"""

import argparse
import datetime
import pathlib

import numpy as np
import astropy.table
import astropy.units as u
import requests
import scipy.io
from sunpy.io.special import read_genx

mirrors = [
    "https://soho.nascom.nasa.gov/solarsoft/sdo/aia/response/",
    "https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response/",
]
"""The SolarSoft mirrors to download the files from, tried in order."""

files = dict(
    area="aia_V9_all_fullinst.genx",
    emissivity="aia_V9_fullemiss.genx",
    chiantifix="aia_V9_chiantifix.genx",
)
"""The files ``aia_get_response`` reads for version 10."""

channels = [94, 131, 171, 193, 211, 304, 335]
"""The EUV channels with thin focal-plane filters, the default of ``aia_get_response``."""

crosstalk = {94: 304, 131: 335, 304: 94, 335: 131}
"""The channel sharing a telescope with each channel which has one."""

output = (
    pathlib.Path(__file__).parents[1] / "sdo/aia/data/aia_V10_temperature_response.ecsv"
)


def download(directory: pathlib.Path, name: str) -> pathlib.Path:
    """A SolarSoft file, downloaded into `directory` unless it is already there."""
    path = directory / name
    if path.exists():
        return path
    for mirror in mirrors:
        try:
            with requests.get(mirror + name, stream=True, timeout=600) as r:
                r.raise_for_status()
                partial = path.with_suffix(".part")
                with open(partial, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            partial.rename(path)
            return path
        except requests.RequestException as e:
            print(f"{mirror + name}: {e}")
    raise RuntimeError(f"could not download {name} from any mirror")


def effective_area(area: dict, channel: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    The wavelengths, effective area in cm^2 DN / photon, and plate scale in
    sr / pix of a channel.

    Ports ``aia_bp_blend_channels`` (the crosstalk from the other channel of
    the telescope) and the ``/dn`` branch of ``aia_bp_parse_effarea``.
    """
    full = area[f"A{channel}_FULL"]
    wave = np.asarray(area[f"A{channel}"]["WAVE"])
    effarea = np.asarray(full["EFFAREA"])
    if channel in crosstalk:
        other = area[f"A{crosstalk[channel]}_FULL"]
        effarea = effarea + (
            other["ENT_FILTER"]
            * other["PRIMARY"]
            * other["SECONDARY"]
            * other["CONTAM"]
            * other["CCD"]
            * other["GEOAREA"]
            * full["FP_FILTER"]
        )
    evperphot = np.float32(12398.0) / wave
    elecperphot = np.maximum(evperphot * full["ELECPEREV"], 1)
    effarea = effarea * elecperphot / full["ELECPERDN"]
    return wave, effarea, float(area[f"A{channel}"]["PLATESCALE"])


def temperature_response(
    wave: np.ndarray,
    effarea: np.ndarray,
    platescale: float,
    emiss_wave: np.ndarray,
    emiss: np.ndarray,
) -> np.ndarray:
    """
    The temperature response of a channel, in DN cm^5 s^-1 pix^-1.

    Ports ``aia_bp_make_tresp``: the effective area interpolated onto the
    wavelengths of the emissivity, zero outside its own, and summed against
    the emissivity at each temperature.
    """
    iresponse = np.interp(emiss_wave, wave, effarea)
    iresponse[(emiss_wave < wave.min()) | (emiss_wave > wave.max())] = 0
    wavestep = emiss_wave[1] - emiss_wave[0]
    return (np.maximum(iresponse, 0) @ emiss) * platescale * wavestep


def main(directory: pathlib.Path, check: None | pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {key: download(directory, name) for key, name in files.items()}

    area = read_genx(paths["area"])
    emissivity = read_genx(paths["emissivity"])
    fix = read_genx(paths["chiantifix"])

    total = emissivity["TOTAL"]
    # Stored in single precision, so 6.6 reads as 6.5999999, which puts its
    # temperature just below 10^6.6 K. Rounded back onto the grid of steps of
    # 0.05 it was made on.
    logte = np.round(np.asarray(total["LOGTE"], dtype=float), 2)
    emiss_wave = np.asarray(total["WAVE"])
    emiss = np.asarray(total["EMISSIVITY"])
    if emiss.shape != (emiss_wave.size, logte.size):
        emiss = emiss.T
    if not np.allclose(fix["LOGTE"], logte):
        raise ValueError("the chiantifix is on a different temperature grid")
    fix_channels = [int(str(c).removeprefix("A")) for c in fix["CHANNELS"]]
    fix_values = np.asarray(fix["EMPIRICAL_MINUS_RAW"])

    unit = u.DN * u.cm**5 / u.s / u.pix
    table = astropy.table.QTable()
    table["temperature"] = 10**logte * u.K
    for channel in channels:
        wave, effarea, platescale = effective_area(area, channel)
        response = temperature_response(wave, effarea, platescale, emiss_wave, emiss)
        table[f"response_{channel}"] = response * unit
    for channel in channels:
        values = np.zeros(logte.size)
        if channel in fix_channels:
            values = fix_values[fix_channels.index(channel)]
        table[f"chiantifix_{channel}"] = values * unit

    general = emissivity["GENERAL"]
    table.meta = dict(
        channels=channels,
        description=(
            "Temperature response of the AIA EUV channels (thin focal-plane "
            "filters, with crosstalk) computed as aia_get_response(/temperature, "
            "/dn) does in SolarSoft version 10, without the time-dependent "
            "corrections. response_N is the response of channel N, and "
            "chiantifix_N the correction /chiantifix adds to the EVE-normalized "
            "response. The emission measure is that of n_e n_H."
        ),
        files=[str(f) for f in files.values()],
        file_dates={
            key: str(read["HEADER"].get("CREATION", ""))
            for key, read in [
                ("area", area),
                ("emissivity", emissivity),
                ("chiantifix", fix),
            ]
        },
        chianti={
            key.lower(): str(general[key])
            for key in general
            if key.upper()
            in {
                "VERSION",
                "ABUNDFILE",
                "IONEQ_NAME",
                "MODEL_NAME",
                "MODEL_PE",
                "MODEL_NE",
                "MODEL_TE",
                "ADD_PROTONS",
                "PHOTOEXCITATION",
                "WVL_LIMITS",
            }
        },
        generated=datetime.date.today().isoformat(),
        generator="tools/aia_temperature_response.py",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.write(output, format="ascii.ecsv", overwrite=True)
    print(f"wrote {output}")

    if check is not None:
        compare(table, check)


def compare(table: astropy.table.QTable, directory: pathlib.Path) -> None:
    """Compare with the ``.dat`` files of ``make_aiaresp_forpy.pro``."""
    import aiapy.calibrate.utils
    import aiapy.response
    import astropy.time

    correction_table = aiapy.calibrate.utils.get_correction_table("SSW")
    for name, eve, fix in [
        ("aia_tresp.dat", False, False),
        ("aia_tresp_en.dat", True, False),
        ("aia_tresp_en_cf.dat", True, True),
    ]:
        idl = scipy.io.readsav(directory / name)
        print(name)
        for j, channel in enumerate(
            int(str(c, "ascii").removeprefix("A")) for c in idl["channels"]
        ):
            ours = table[f"response_{channel}"].value
            if eve:
                # the same in every epoch, so at the start of the first
                aia = aiapy.response.Channel(channel * u.AA)
                time = astropy.time.Time(correction_table["T_START"].min())
                ours = ours * aia.eve_correction(time, correction_table).value
            if fix:
                ours = ours + table[f"chiantifix_{channel}"].value
            theirs = idl["tr"][j]
            where = theirs > 1e-3 * theirs.max()
            error = np.abs(ours[where] / theirs[where] - 1).max()
            print(
                f"  {channel:>3}: largest relative difference {error:.1e} above 1e-3 of the peak"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--check", type=pathlib.Path, default=None)
    args = parser.parse_args()
    main(args.directory, args.check)
