from typing import cast
import pytest
import numpy as np
import astropy.units as u
import aiapy.calibrate
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
    ],
)
def test_uncertainty(
    counts: na.ScalarArray,
    wavelength: u.Quantity | na.ScalarArray,
) -> None:
    result = sdo.aia.uncertainty(counts, wavelength)

    assert result.shape == counts.shape
    wavelength = na.as_named_array(wavelength)
    for index in na.ndindex(counts.shape):
        channel = cast(na.ScalarArray, wavelength[index]).ndarray
        value = np.maximum(cast(na.ScalarArray, counts[index]).ndarray, 0 * u.DN)
        expected = aiapy.calibrate.estimate_error(value / u.pix, channel) * u.pix
        assert np.all(np.isfinite(expected))
        found = cast(na.ScalarArray, result[index]).ndarray
        assert np.allclose(found, expected, rtol=1e-12, atol=0)
