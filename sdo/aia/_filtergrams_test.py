from typing import cast
import pathlib
import pytest
import numpy as np
import astropy.units as u
import astropy.io.fits
import astropy.wcs
import named_arrays as na
import sdo


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        sdo.aia.open("2021-09-23T06:00"),
        sdo.aia.open(
            time_start="2021-09-23T06:00",
            wavelength=na.ScalarArray([304] * u.AA, "wavelength"),
            limit=1,
        ),
    ],
)
class TestSpectrographObservation:

    def test_axis_time(self, array: sdo.aia.Filtergram):
        assert isinstance(array.axis_time, str)

    def test_axis_wavelength(self, array: sdo.aia.Filtergram):
        assert isinstance(array.axis_wavelength, str)

    def test_axis_detector_x(self, array: sdo.aia.Filtergram):
        assert isinstance(array.axis_detector_x, str)

    def test_axis_detector_y(self, array: sdo.aia.Filtergram):
        assert isinstance(array.axis_detector_y, str)

    def test_timedelta(self, array: sdo.aia.Filtergram) -> None:
        timedelta = cast(na.ScalarArray, na.as_named_array(array.timedelta))
        shape = {
            array.axis_time: array.outputs.shape[array.axis_time],
            array.axis_wavelength: array.outputs.shape[array.axis_wavelength],
        }
        assert timedelta.shape == shape
        assert np.all(timedelta.ndarray_aligned(tuple(shape)) > 0 * u.s)


def _fits(path: pathlib.Path, num: int, exptime: float) -> pathlib.Path:
    """A small FITS image with the Sun-centered WCS registration leaves."""
    header = astropy.io.fits.Header()
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = (num + 1) / 2
    header["CRPIX2"] = (num + 1) / 2
    header["CRVAL1"] = 100.0
    header["CRVAL2"] = 200.0
    header["CDELT1"] = 0.6
    header["CDELT2"] = 0.6
    header["DATE-OBS"] = "2024-05-10T17:59:47"
    header["EXPTIME"] = exptime
    data = np.arange(num * num, dtype=float).reshape(num, num)
    astropy.io.fits.PrimaryHDU(data, header).writeto(path)
    return path


def test_from_fits(tmp_path: pathlib.Path) -> None:
    """
    Images of different sizes, as registration leaves them, are centered in
    the array of the first, and every vertex has the coordinates the WCS of
    its file gives it.
    """
    paths = [
        _fits(tmp_path / "a.fits", num=8, exptime=2.9),
        _fits(tmp_path / "b.fits", num=6, exptime=2.0),
    ]
    files = na.ScalarArray(np.array([paths], dtype=object), axes=("t", "w"))
    images = sdo.aia.Filtergram.from_fits(
        path=files,
        wavelength=na.ScalarArray([171, 193] * u.AA, axes="w"),
        axis_time="t",
        axis_wavelength="w",
    )

    assert images.outputs.shape == dict(t=1, w=2, detector_y=8, detector_x=8)
    timedelta = cast(na.ScalarArray, images.timedelta)
    assert np.all(timedelta.ndarray == [[2.9, 2.0]] * u.s)

    smaller = cast(na.ScalarArray, images.outputs[dict(t=0, w=1)])
    data = smaller.ndarray_aligned(("detector_y", "detector_x")).value
    assert np.all(np.isnan(data[0])) and np.all(np.isnan(data[:, -1]))
    assert np.array_equal(data[1:-1, 1:-1], np.arange(36.0).reshape(6, 6))

    for i, (path, offset) in enumerate(zip(paths, [0, 1])):
        wcs = astropy.wcs.WCS(astropy.io.fits.getheader(path))
        # the lower left corner of the first pixel of the file
        x, y = wcs.pixel_to_world_values(-0.5, -0.5)
        index = dict(t=0, w=i, detector_x=offset, detector_y=offset)
        position = images.inputs.position[index]
        found_x = cast(na.ScalarArray, position.x).ndarray
        found_y = cast(na.ScalarArray, position.y).ndarray
        assert np.isclose(found_x, x * u.deg, rtol=0, atol=1e-3 * u.arcsec)
        assert np.isclose(found_y, y * u.deg, rtol=0, atol=1e-3 * u.arcsec)
