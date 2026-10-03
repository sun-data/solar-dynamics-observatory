from typing import cast
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import aiapy.calibrate
import aiapy.calibrate.utils
import aiapy.response
import named_arrays as na
import sdo

_correction_table = aiapy.calibrate.utils.get_correction_table("SSW")

_unit = u.DN * u.cm**5 / u.s / u.pix


def _values(a: na.AbstractArray) -> np.ndarray:
    """The values of a scalar array, with its axes in the order they are in."""
    a = cast(na.AbstractScalarArray, a)
    return a.ndarray_aligned(tuple(a.shape))


@pytest.mark.parametrize(
    argnames="wavelength,shape",
    argvalues=[
        ([94, 131, 171, 193, 211, 335] * u.AA, dict(temperature=101, wavelength=6)),
        (304 * u.AA, dict(temperature=101, wavelength=1)),
        (
            na.ScalarArray([171, 193] * u.AA, axes="channel"),
            dict(temperature=101, channel=2),
        ),
    ],
)
def test_temperature_response(
    wavelength: u.Quantity | na.ScalarArray,
    shape: dict[str, int],
) -> None:
    result = sdo.aia.temperature_response(wavelength)
    assert result.outputs.shape == shape
    assert result.inputs.shape == dict(temperature=101)
    assert na.unit(result.outputs) == _unit
    assert np.all(_values(result.outputs) >= 0)


@pytest.mark.parametrize(
    argnames="channel,logte",
    argvalues=[
        # the temperature of the main peak of each channel, from Table 1 of
        # Boerner et al. (2012)
        (171, 5.8),
        (193, 6.2),
        (211, 6.3),
        (335, 6.4),
        (94, 6.8),
    ],
)
def test_temperature_response_peak(channel: int, logte: float) -> None:
    result = sdo.aia.temperature_response(channel * u.AA)
    temperature = result.inputs.ndarray
    response = _values(result.outputs)[:, 0]
    # the hot peaks of 94 and 335, not their cool ones
    hot = temperature > (10**5.9 if channel in (94, 335) else 0) * u.K
    peak = temperature[hot][np.argmax(response[hot])]
    assert np.abs(np.log10(peak.to_value(u.K)) - logte) < 0.15


def test_temperature_response_corrections() -> None:
    """
    The corrections scale the response as ``aia_get_response`` scales it, and
    the empirical correction to CHIANTI changes only the 94 angstrom channel.
    """
    channels = [94, 131, 171, 193, 211, 304, 335] * u.AA
    time = astropy.time.Time("2024-05-10T18:00")
    kwargs = dict(wavelength=channels, correction_table=_correction_table)

    base = sdo.aia.temperature_response(**kwargs).outputs
    eve = sdo.aia.temperature_response(eve=True, time=time, **kwargs).outputs
    fix = sdo.aia.temperature_response(
        eve=True, chiantifix=True, time=time, **kwargs
    ).outputs

    for i, channel in enumerate(channels):
        index = dict(wavelength=i)
        aia = aiapy.response.Channel(channel)
        scale = aia.eve_correction(
            time, _correction_table
        ) * aiapy.calibrate.degradation(
            channel, time, correction_table=_correction_table
        )
        expected = _values(base[index]) * scale
        assert np.allclose(_values(eve[index]), expected, rtol=1e-12, atol=0)
        changed = not np.allclose(
            _values(fix[index]), _values(eve[index]), rtol=1e-6, atol=0
        )
        assert changed == (channel == 94 * u.AA)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(wavelength=1600 * u.AA),
        dict(wavelength=na.ScalarArray([[171]] * u.AA, axes=("a", "b"))),
        dict(chiantifix=True),
    ],
)
def test_temperature_response_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        sdo.aia.temperature_response(**kwargs)
