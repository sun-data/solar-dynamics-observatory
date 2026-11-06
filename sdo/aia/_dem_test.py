import pytest
import numpy as np
import astropy.units as u
import astropy.time
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
        dict(emissivity="CHIANTI 11"),
    ],
)
def test_dem_response(kwargs: dict) -> None:
    """Check the response is the one asked for."""
    calls = []

    def plowman(**k: object) -> tuple[na.FunctionArray, na.ScalarArray]:
        """Stand in for the inversion, remembering what it was given."""
        calls.append(k)
        return na.FunctionArray(None, None), na.ScalarArray(0)

    filtergram = _filtergram()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(utu.dem, "plowman", plowman)
        sdo.aia.dem(filtergram, axis_temperature="t", **kwargs)

    (call,) = calls
    temperature_min = kwargs.get("temperature_min", 10**5.5 * u.K)
    temperature_max = kwargs.get("temperature_max", 10**7.5 * u.K)
    response = sdo.aia.temperature_response(
        wavelength=_channels,
        time=_time,
        eve=kwargs.get("eve", True),
        axis_wavelength="channel",
        axis_temperature="t",
        emissivity=kwargs.get("emissivity", "SolarSoft"),
    )
    t = response.inputs
    response = response[(t >= temperature_min) & (t <= temperature_max)]
    assert np.all(call["response"].inputs == response.inputs)
    assert np.all(call["response"].outputs == response.outputs)


def test_dem_uncertainty() -> None:
    """Check an uncertainty given is used in place of the default."""
    calls = []

    def plowman(**k: object) -> tuple[na.FunctionArray, na.ScalarArray]:
        """Stand in for the inversion, remembering what it was given."""
        calls.append(k)
        return na.FunctionArray(None, None), na.ScalarArray(0)

    filtergram = _filtergram()
    uncertainty = 3 * sdo.aia.uncertainty(filtergram.outputs, _channels)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(utu.dem, "plowman", plowman)
        sdo.aia.dem(filtergram, uncertainty=uncertainty)

    (call,) = calls
    expected = uncertainty / _timedelta / u.pix
    assert np.all(call["uncertainty"] == expected)
    assert call["axis_channel"] == "channel"
