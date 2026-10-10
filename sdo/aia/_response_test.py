from typing import Any, cast
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import astropy.table
import aiapy.calibrate
import aiapy.calibrate.utils
import aiapy.response
import named_arrays as na
import sdo

_correction_table = aiapy.calibrate.utils.get_correction_table("SSW")

_unit = u.DN * u.cm**5 / u.s / u.pix

_emissivities = ["SolarSoft", "CHIANTI 11"]


def _values(a: na.AbstractArray) -> np.ndarray:
    """The values of a scalar array, with its axes in the order they are in."""
    a = cast(na.AbstractScalarArray, a)
    return a.ndarray_aligned(tuple(a.shape))


@pytest.mark.parametrize("emissivity", _emissivities)
@pytest.mark.parametrize(
    argnames="wavelength,shape",
    argvalues=[
        ([94, 131, 171, 193, 211, 335] * u.AA, dict(temperature=101, wavelength=6)),
        (304 * u.AA, dict(temperature=101, wavelength=1)),
        (na.ScalarArray(171 * u.AA), dict(temperature=101, wavelength=1)),
        (
            na.ScalarArray([171, 193] * u.AA, axes="channel"),
            dict(temperature=101, channel=2),
        ),
    ],
)
def test_temperature_response(
    wavelength: u.Quantity | na.ScalarArray,
    shape: dict[str, int],
    emissivity: Any,
) -> None:
    result = sdo.aia.temperature_response(wavelength, emissivity=emissivity)
    assert result.outputs.shape == shape
    assert result.inputs.shape == dict(temperature=101)
    assert na.unit(result.outputs) == _unit
    assert np.all(_values(result.outputs) >= 0)


@pytest.mark.parametrize("emissivity", _emissivities)
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
def test_temperature_response_peak(
    channel: int,
    logte: float,
    emissivity: Any,
) -> None:
    result = sdo.aia.temperature_response(channel * u.AA, emissivity=emissivity)
    temperature = result.inputs.ndarray
    response = _values(result.outputs[dict(wavelength=0)])
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


def test_temperature_response_corrections_chianti() -> None:
    """
    The corrections are to the effective area, so they scale the response
    from CHIANTI 11 by what they scale the response of SolarSoft by.
    """
    channels = [94, 131, 171, 193, 211, 304, 335] * u.AA
    time = astropy.time.Time("2024-05-10T18:00")
    kwargs = dict(wavelength=channels, correction_table=_correction_table)

    ssw = sdo.aia.temperature_response(**kwargs).outputs
    ssw_eve = sdo.aia.temperature_response(eve=True, time=time, **kwargs).outputs
    new = sdo.aia.temperature_response(emissivity="CHIANTI 11", **kwargs).outputs
    new_eve = sdo.aia.temperature_response(
        eve=True, time=time, emissivity="CHIANTI 11", **kwargs
    ).outputs

    assert not np.allclose(_values(new), _values(ssw), rtol=1e-2, atol=0)
    for i, _ in enumerate(channels):
        index = dict(wavelength=i)
        scale = _values(ssw_eve[index])[50] / _values(ssw[index])[50]
        expected = _values(new[index]) * scale
        assert np.allclose(_values(new_eve[index]), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("channel", [94, 131, 171, 193, 211, 304, 335])
def test_temperature_response_chianti_solarsoft(channel: int) -> None:
    """
    The response from CHIANTI 11 agrees with that of SolarSoft where the
    atomic data has changed little: within 10% at the peak of each channel,
    which the boost of He II dominates in 304 angstroms, and within 2% at
    log T 8, where the free-free continuum dominates.
    """
    ssw = sdo.aia.temperature_response(channel * u.AA)
    new = sdo.aia.temperature_response(channel * u.AA, emissivity="CHIANTI 11")
    logt = np.log10(_values(ssw.inputs).to_value(u.K))
    ssw = _values(ssw.outputs[dict(wavelength=0)]).value
    new = _values(new.outputs[dict(wavelength=0)]).value

    assert 0.9 < new.max() / ssw.max() < 1.1
    hot = np.argmin(np.abs(logt - 8))
    assert 0.98 < new[hot] / ssw[hot] < 1.02


def test_temperature_response_positional() -> None:
    """The arguments of version 1.2.0 keep their positions."""
    result = sdo.aia.temperature_response(
        [171] * u.AA, None, False, False, "channel", "logte"
    )
    assert result.outputs.shape == dict(logte=101, channel=1)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(wavelength=1600 * u.AA),
        dict(wavelength=na.ScalarArray([[171]] * u.AA, axes=("a", "b"))),
        dict(chiantifix=True),
        dict(chiantifix=True, eve=True, emissivity="CHIANTI 11"),
        dict(emissivity="CHIANTI 10"),
    ],
)
def test_temperature_response_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        sdo.aia.temperature_response(**kwargs)


@pytest.mark.parametrize(
    argnames="time",
    argvalues=[
        ["2024-05-10", "2024-05-11"],
        # as many as the temperatures, which would multiply them one by one
        astropy.time.Time("2024-05-10") + np.arange(101) * u.day,
    ],
)
def test_temperature_response_times(time: list[str] | astropy.time.Time) -> None:
    with pytest.raises(ValueError, match="single time"):
        sdo.aia.temperature_response(171 * u.AA, time=time)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        # each alone, since together the area of the first epoch, which a
        # table of every version can take from another version, cancels
        dict(eve=True),
        dict(time="2024-05-10T18:00"),
    ],
)
def test_temperature_response_version(kwargs: dict[str, Any]) -> None:
    """
    Only the rows of version 10 of a correction table are used, so that a
    table of every version, like the one from the JSOC, gives what the table
    of version 10 does, and a table without version 10 is an error.
    """
    other = _correction_table.copy()
    other["VER_NUM"] = 9
    other["EFF_AREA"] = 2 * other["EFF_AREA"]
    # older, so that it comes first when the rows are sorted by date
    other["DATE"] = "2012-01-01T00:00:00.000"
    both = astropy.table.vstack([other, _correction_table])

    kwargs = dict(wavelength=[94, 171] * u.AA, **kwargs)
    expected = sdo.aia.temperature_response(
        correction_table=_correction_table,
        **kwargs,
    )
    found = sdo.aia.temperature_response(correction_table=both, **kwargs)
    assert np.all(_values(found.outputs) == _values(expected.outputs))

    with pytest.raises(ValueError, match="version 10"):
        sdo.aia.temperature_response(correction_table=other, **kwargs)


def test_temperature_response_default_table() -> None:
    """
    Without a correction table, the version 10 table SolarSoft uses is
    fetched, and gives what passing it does.
    """
    time = astropy.time.Time("2024-05-10T18:00")
    found = sdo.aia.temperature_response(171 * u.AA, time=time, eve=True)
    expected = sdo.aia.temperature_response(
        171 * u.AA,
        time=time,
        eve=True,
        correction_table=_correction_table,
    )
    assert np.all(_values(found.outputs) == _values(expected.outputs))


def test_temperature_response_copy() -> None:
    """
    Changing a response in place leaves the next one alone, so neither shares
    the table they are read from.
    """
    result = sdo.aia.temperature_response(171 * u.AA)
    inputs = _values(result.inputs).copy()
    outputs = _values(result.outputs).copy()

    cast(na.ScalarArray, result.inputs).ndarray[:] = 0 * u.K
    cast(na.ScalarArray, result.outputs).ndarray[:] = 0 * _unit

    found = sdo.aia.temperature_response(171 * u.AA)
    assert np.all(_values(found.inputs) == inputs)
    assert np.all(_values(found.outputs) == outputs)


def test_temperature_response_eve_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Without a time, the EVE normalization does not depend on the current
    date, which falls in no epoch once the last one ends.
    """
    expected = sdo.aia.temperature_response(171 * u.AA, eve=True)

    def now(cls: type[astropy.time.Time]) -> astropy.time.Time:
        return cls("2031-01-01")

    monkeypatch.setattr(astropy.time.Time, "now", classmethod(now))
    assert astropy.time.Time.now() > _correction_table["T_STOP"].max()
    found = sdo.aia.temperature_response(171 * u.AA, eve=True)
    assert np.all(_values(found.outputs) == _values(expected.outputs))


@pytest.mark.parametrize("emissivity", _emissivities)
def test_temperature_response_grid(emissivity: Any) -> None:
    """
    The temperatures are exactly :math:`10^{4 + 0.05 i}` kelvin, so that their
    logarithms fall on the edges of ranges like 6.3 to 6.6, rather than just
    below them as the single-precision grid of SolarSoft does.
    """
    result = sdo.aia.temperature_response(171 * u.AA, emissivity=emissivity)
    logt = np.log10(_values(result.inputs).to_value(u.K))
    assert np.all(logt == np.round(np.linspace(4, 9, 101), 2))


@pytest.mark.parametrize(
    argnames="kwargs,reference",
    argvalues=[
        (
            dict(),
            {
                94: [
                    7.989764e-29,
                    1.293278e-27,
                    7.934313e-28,
                    2.082041e-27,
                    5.232063e-28,
                ],
                131: [
                    1.146029e-26,
                    2.403140e-26,
                    1.521314e-27,
                    5.886691e-28,
                    1.372394e-26,
                ],
                171: [
                    6.538093e-26,
                    1.249941e-24,
                    2.393434e-26,
                    1.054749e-27,
                    3.964249e-28,
                ],
                193: [
                    7.151754e-26,
                    1.562612e-25,
                    1.829818e-25,
                    6.377391e-27,
                    8.007815e-27,
                ],
                211: [
                    2.106046e-26,
                    3.746260e-26,
                    1.628607e-25,
                    2.620222e-27,
                    3.733237e-28,
                ],
                335: [
                    3.305645e-27,
                    3.847616e-27,
                    3.793753e-27,
                    1.570354e-27,
                    4.659536e-28,
                ],
            },
        ),
        (
            dict(eve=True, chiantifix=True),
            {
                94: [
                    6.301737e-29,
                    1.549968e-27,
                    1.242555e-27,
                    1.518839e-27,
                    3.850719e-28,
                ],
                131: [
                    1.196602e-26,
                    2.509188e-26,
                    1.588448e-27,
                    6.146466e-28,
                    1.432957e-26,
                ],
                171: [
                    6.329041e-26,
                    1.209975e-24,
                    2.316905e-26,
                    1.021024e-27,
                    3.837494e-28,
                ],
                193: [
                    8.619969e-26,
                    1.883408e-25,
                    2.205470e-25,
                    7.686633e-27,
                    9.651774e-27,
                ],
                211: [
                    2.282896e-26,
                    4.060842e-26,
                    1.765365e-25,
                    2.840248e-27,
                    4.046726e-28,
                ],
                335: [
                    4.207437e-27,
                    4.897260e-27,
                    4.828703e-27,
                    1.998752e-27,
                    5.930676e-28,
                ],
            },
        ),
    ],
)
def test_temperature_response_idl(
    kwargs: dict,
    reference: dict[int, list[float]],
) -> None:
    """
    The response agrees with ``aia_get_response(/temperature, /dn)`` in IDL,
    plain and with ``/evenorm, /chiantifix``, at log temperatures of 5.5, 5.9,
    6.3, 6.7 and 7.1. The values are from ``aia_tresp.dat`` and
    ``aia_tresp_en_cf.dat``, which ``make_aiaresp_forpy.pro`` of ``demreg``
    writes, to the seven digits of their single precision.
    """
    channels = list(reference)
    result = sdo.aia.temperature_response(
        wavelength=channels * u.AA,
        correction_table=_correction_table,
        **kwargs,
    )
    logt = np.log10(_values(result.inputs).to_value(u.K))
    index = [np.argmin(np.abs(logt - t)) for t in [5.5, 5.9, 6.3, 6.7, 7.1]]
    for i, channel in enumerate(channels):
        found = _values(result.outputs[dict(wavelength=i)])[index]
        expected = reference[channel] * _unit
        assert np.allclose(found, expected, rtol=1e-5, atol=0)
