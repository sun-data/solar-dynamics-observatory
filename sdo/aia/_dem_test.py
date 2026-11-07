import inspect
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import aiapy.calibrate.utils
import named_arrays as na
import utu
import sdo

_channels = na.ScalarArray([94, 131, 171, 193, 211, 335] * u.AA, axes="channel")
"""The six optically thin EUV channels, along an axis not named the default."""

_time = astropy.time.Time("2016-07-15T00:00:00")
"""The time of the earliest image of :func:`_filtergram`."""

_timedelta = 2 * u.s
"""The exposure time of every image of :func:`_filtergram`."""


def _response() -> na.FunctionArray:
    """The response the images of :func:`_filtergram` are computed with."""
    return sdo.aia.temperature_response(
        wavelength=_channels,
        time=_time,
        eve=True,
        axis_wavelength="channel",
        axis_temperature="t",
    )


def _filtergram() -> sdo.aia.Filtergram:
    """
    AIA images three pixels by two, in each of the six channels, of plasma
    whose DEM is a Gaussian in :math:`\\log_{10} T` around 2 MK, more of it
    in each pixel than in the last.

    The images are the counts that DEM gives through the response, without
    noise. They are taken a second apart, the last channel first, so that
    the earliest image is not the first.
    """
    response = _response()
    temperature = response.inputs
    logt = np.log10(temperature.to(u.K).value)
    amplitude = na.ScalarArray(np.array([[1, 2], [3, 4], [5, 6]]), ("x", "y"))
    dem = 1e27 * amplitude * np.exp(-np.square((logt - 6.3) / 0.15) / 2) / u.cm**5
    dem = na.FunctionArray(temperature, dem)
    em = utu.dem.emission_measure(dem, "t")
    rate = (response.outputs * em).sum("t")
    counts = (rate * _timedelta * u.pix).to(u.DN)

    time = _time + np.arange(_channels.size)[::-1] * u.s
    position = na.Cartesian2dVectorLinearSpace(
        start=-0.9 * u.arcsec,
        stop=0.9 * u.arcsec,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=na.Cartesian2dVectorArray(4, 3),
    ).explicit
    return sdo.aia.Filtergram(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=na.ScalarArray(time, axes="channel"),
            wavelength=_channels,
            position=position,
        ),
        outputs=counts,
        axis_wavelength="channel",
        axis_detector_x="x",
        axis_detector_y="y",
        timedelta=_timedelta,
    )


def _plowman_call(**kwargs: object) -> dict[str, object]:
    """
    Call :func:`sdo.aia.dem` on :func:`_filtergram` with the inversion
    stubbed out, and return the arguments the inversion was given.
    """
    calls = []

    def plowman(**k: object) -> tuple[na.FunctionArray, na.ScalarArray]:
        """Stand in for the inversion, remembering what it was given."""
        calls.append(k)
        return na.FunctionArray(None, None), na.ScalarArray(0)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(utu.dem, "plowman", plowman)
        sdo.aia.dem(_filtergram(), **kwargs)

    (call,) = calls
    return call


def test_dem() -> None:
    """
    Check the DEM is the inversion of the intensities, with the uncertainty
    of :func:`sdo.aia.uncertainty`, against the response at the time of the
    earliest image, cut to the default range of temperature.
    """
    filtergram = _filtergram()

    dem, chi2 = sdo.aia.dem(filtergram, axis_temperature="t")

    response = _response()
    t = response.inputs
    response = response[(t >= 10**5.5 * u.K) & (t <= 10**7.5 * u.K)]
    uncertainty = sdo.aia.uncertainty(filtergram.outputs, _channels)
    expected, chi2_expected = utu.dem.plowman(
        intensity=filtergram.outputs / _timedelta / u.pix,
        uncertainty=uncertainty / _timedelta / u.pix,
        response=response,
        axis_channel="channel",
        axis_temperature="t",
    )

    assert np.all(dem.inputs == expected.inputs)
    assert np.all(dem.outputs == expected.outputs)
    assert np.all(chi2 == chi2_expected)

    logt = np.log10(dem.inputs.to(u.K).value)
    assert np.isclose(logt.min().ndarray, 5.5)
    assert np.isclose(logt.max().ndarray, 7.5)
    assert na.shape(dem.outputs) == dict(t=logt.size, x=3, y=2)
    assert dem.outputs.unit.is_equivalent(u.cm**-5)

    # The images have no noise, so the DEM reproduces them to within their
    # uncertainty, which is what the inversion aims at.
    assert np.all(chi2 >= 0)
    assert np.all(chi2 <= 1.1)
    em = utu.dem.emission_measure(dem, "t")
    rate = (response.outputs * em).sum("t")
    counts = rate * _timedelta * u.pix
    error = sdo.aia.uncertainty(filtergram.outputs, _channels)
    assert np.all(np.abs(counts - filtergram.outputs) < 3 * error)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(temperature_min=10**6 * u.K, temperature_max=10**7 * u.K),
        dict(eve=False),
        dict(chiantifix=True),
        dict(emissivity="CHIANTI 11"),
    ],
)
def test_dem_response(kwargs: dict) -> None:
    """Check the response is the one asked for."""
    call = _plowman_call(axis_temperature="t", **kwargs)

    temperature_min = kwargs.get("temperature_min", 10**5.5 * u.K)
    temperature_max = kwargs.get("temperature_max", 10**7.5 * u.K)
    response = sdo.aia.temperature_response(
        wavelength=_channels,
        time=_time,
        eve=kwargs.get("eve", True),
        chiantifix=kwargs.get("chiantifix", False),
        axis_wavelength="channel",
        axis_temperature="t",
        emissivity=kwargs.get("emissivity", "SolarSoft"),
    )
    t = response.inputs
    response = response[(t >= temperature_min) & (t <= temperature_max)]
    assert np.all(call["response"].inputs == response.inputs)
    assert np.all(call["response"].outputs == response.outputs)


def test_dem_correction_table() -> None:
    """Check a correction table given is the one the response is corrected with."""
    table = aiapy.calibrate.utils.get_correction_table("SSW")
    table["EFF_AREA"] *= 2

    call = _plowman_call(axis_temperature="t", correction_table=table)

    response = sdo.aia.temperature_response(
        wavelength=_channels,
        time=_time,
        eve=True,
        axis_wavelength="channel",
        axis_temperature="t",
        correction_table=table,
    )
    t = response.inputs
    response = response[(t >= 10**5.5 * u.K) & (t <= 10**7.5 * u.K)]
    assert np.all(call["response"].outputs == response.outputs)

    # the table given changes the response, so it is not the default
    default = _plowman_call(axis_temperature="t")
    assert np.all(call["response"].outputs != default["response"].outputs)


def test_dem_uncertainty() -> None:
    """Check an uncertainty given is used in place of the default."""
    filtergram = _filtergram()
    uncertainty = 3 * sdo.aia.uncertainty(filtergram.outputs, _channels)

    call = _plowman_call(uncertainty=uncertainty)

    expected = uncertainty / _timedelta / u.pix
    assert np.all(call["uncertainty"] == expected)
    assert call["axis_channel"] == "channel"


_plowman_arguments = [
    "smoothness",
    "chi2_target",
    "tolerance",
    "steps",
    "iterations_max",
    "iterations_min",
    "floor",
]
"""The arguments of :func:`utu.dem.plowman` that :func:`sdo.aia.dem` passes on."""


def test_dem_plowman_defaults() -> None:
    """Check the arguments passed to the inversion default to its own defaults."""
    dem = inspect.signature(sdo.aia.dem).parameters
    plowman = inspect.signature(utu.dem.plowman).parameters
    for name in _plowman_arguments:
        assert dem[name].default == plowman[name].default

    call = _plowman_call()
    for name in _plowman_arguments:
        assert call[name] == plowman[name].default


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(smoothness=4),
        dict(chi2_target=0.5),
        dict(tolerance=0.01),
        dict(steps=(0.2, 0.75)),
        dict(iterations_max=10),
        dict(iterations_min=2),
        dict(floor=1 * u.DN / u.s / u.pix),
    ],
)
def test_dem_plowman(kwargs: dict) -> None:
    """Check the arguments of the inversion given are passed on to it."""
    call = _plowman_call(**kwargs)
    for name, value in kwargs.items():
        assert call[name] == value
