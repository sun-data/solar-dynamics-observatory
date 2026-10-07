from typing import cast
import pytest
import numpy as np
import astropy.units as u
import astropy.table
import aiapy.calibrate
import aiapy.calibrate.utils
import named_arrays as na
import sdo


@pytest.mark.parametrize(
    argnames="counts,wavelength",
    argvalues=[
        (
            na.ScalarArray([[0, 10, 1000], [5, 50, 5000]] * u.DN, axes=("w", "x")),
            na.ScalarArray([171, 94] * u.AA, axes="w"),
        ),
        # one channel for every image, with the negative counts of noise
        (
            na.ScalarArray([[0, -3, 1000], [5, 50, 5000]] * u.DN, axes=("t", "x")),
            335 * u.AA,
        ),
        # a single value for each channel, like the total of a region
        (
            na.ScalarArray([1000, 200] * u.DN, axes="w"),
            na.ScalarArray([171, 94] * u.AA, axes="w"),
        ),
        (
            na.ScalarArray(1000 * u.DN),
            171 * u.AA,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="n_sample",
    argvalues=[1, 16],
)
def test_uncertainty(
    counts: na.ScalarArray,
    wavelength: u.Quantity | na.ScalarArray,
    n_sample: int,
) -> None:
    result = sdo.aia.uncertainty(counts, wavelength, n_sample=n_sample)

    assert result.shape == counts.shape
    wavelength = na.as_named_array(wavelength)
    for index in na.ndindex(counts.shape):
        channel = cast(na.ScalarArray, wavelength[index]).ndarray
        value = np.maximum(cast(na.ScalarArray, counts[index]).ndarray, 0 * u.DN)
        expected = aiapy.calibrate.estimate_error(
            value / u.pix,
            channel,
            n_sample=n_sample,
        )
        expected = expected * u.pix
        assert np.all(np.isfinite(expected))
        found = cast(na.ScalarArray, result[index]).ndarray
        assert np.allclose(found, expected, rtol=1e-12, atol=0)


def test_uncertainty_error_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The table of the noise of each channel is read once, rather than once
    for every channel.
    """
    original = aiapy.calibrate.utils.get_error_table
    calls = []

    def get_error_table(**kwargs) -> astropy.table.QTable:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(aiapy.calibrate.utils, "get_error_table", get_error_table)
    sdo.aia._uncertainty._error_table.cache_clear()

    counts = na.ScalarArray([[0, 10, 1000], [5, 50, 5000]] * u.DN, axes=("w", "x"))
    wavelength = na.ScalarArray([171, 94] * u.AA, axes="w")
    try:
        sdo.aia.uncertainty(counts, wavelength)
        sdo.aia.uncertainty(counts, wavelength)
        assert len(calls) == 1
    finally:
        sdo.aia._uncertainty._error_table.cache_clear()
