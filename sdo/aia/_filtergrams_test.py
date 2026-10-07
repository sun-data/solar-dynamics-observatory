from typing import cast
import pathlib
import dataclasses
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


@pytest.mark.parametrize(
    argnames="num,offset",
    argvalues=[
        (6, 1),
        # one pixel smaller, which leaves a row and a column uncovered
        (7, 0),
        (8, 0),
    ],
)
def test_from_fits(tmp_path: pathlib.Path, num: int, offset: int) -> None:
    """
    Images of different sizes, as registration leaves them, are centered in
    the array of the first, the pixels they do not cover are NaN, and every
    vertex has the coordinates the WCS of its file gives it.
    """
    paths = [
        _fits(tmp_path / "a.fits", num=8, exptime=2.9),
        _fits(tmp_path / "b.fits", num=num, exptime=2.0),
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

    second = cast(na.ScalarArray, images.outputs[dict(t=0, w=1)])
    data = second.ndarray_aligned(("detector_y", "detector_x")).value
    inside = (slice(offset, offset + num), slice(offset, offset + num))
    assert np.array_equal(
        data[inside], np.arange(num * num, dtype=float).reshape(num, num)
    )
    outside = np.ones(data.shape, dtype=bool)
    outside[inside] = False
    assert np.all(np.isnan(data[outside]))

    for i, (path, shift) in enumerate(zip(paths, [0, offset])):
        wcs = astropy.wcs.WCS(astropy.io.fits.getheader(path))
        # the lower left corner of the first pixel of the file
        x, y = wcs.pixel_to_world_values(-0.5, -0.5)
        index = dict(t=0, w=i, detector_x=shift, detector_y=shift)
        position = images.inputs.position[index]
        found_x = cast(na.ScalarArray, position.x).ndarray
        found_y = cast(na.ScalarArray, position.y).ndarray
        assert np.isclose(found_x, x * u.deg, rtol=0, atol=1e-3 * u.arcsec)
        assert np.isclose(found_y, y * u.deg, rtol=0, atol=1e-3 * u.arcsec)


def test_from_fits_closes(tmp_path: pathlib.Path) -> None:
    """
    The files are closed once they are read, so they can be deleted, which
    Windows refuses while a file is open.
    """
    paths = [
        _fits(tmp_path / "a.fits", num=8, exptime=2.9),
        _fits(tmp_path / "b.fits", num=6, exptime=2.0),
    ]
    images = sdo.aia.Filtergram.from_fits(
        path=na.ScalarArray(np.array([paths], dtype=object), axes=("t", "w")),
        wavelength=na.ScalarArray([171, 193] * u.AA, axes="w"),
        axis_time="t",
        axis_wavelength="w",
    )
    for path in paths:
        path.unlink()
    assert not np.any(np.isnan(images.outputs[dict(w=0)]))


@pytest.mark.parametrize(
    argnames="item",
    argvalues=[
        dict(t=0),
        dict(w=1, detector_x=slice(2, 6), detector_y=slice(1, 7)),
        dict(t=0, w=slice(0, 2), detector_x=slice(None, 5), detector_y=slice(4, None)),
        dict(detector_x=slice(-3, -1), detector_y=slice(-4, None)),
        # an axis the images do not have is ignored
        dict(t=0, detector_x=slice(3, 5), other=2),
        dict(w=na.ScalarArray(np.array([1, 0]), axes="w")),
    ],
)
def test_getitem(tmp_path: pathlib.Path, item: dict) -> None:
    """
    Selecting and cropping images on the WCS gives the coordinates and the
    data which computing every coordinate first does, and the exposure times
    of the images selected.
    """
    paths = [
        _fits(tmp_path / "a.fits", num=8, exptime=2.9),
        _fits(tmp_path / "b.fits", num=6, exptime=2.0),
    ]
    images = sdo.aia.Filtergram.from_fits(
        path=na.ScalarArray(np.array([paths], dtype=object), axes=("t", "w")),
        wavelength=na.ScalarArray([171, 193] * u.AA, axes="w"),
        axis_time="t",
        axis_wavelength="w",
    )

    found = images[item]
    expected = dataclasses.replace(images, inputs=images.inputs.explicit)[item]

    assert isinstance(found, sdo.aia.Filtergram)
    assert isinstance(found.inputs, na.AbstractWcsVector)
    axes = tuple(expected.outputs.shape)
    assert found.outputs.shape == expected.outputs.shape
    assert np.array_equal(
        cast(na.ScalarArray, found.outputs).ndarray_aligned(axes),
        cast(na.ScalarArray, expected.outputs).ndarray_aligned(axes),
        equal_nan=True,
    )
    position_found = found.inputs.position
    position_expected = expected.inputs.position
    axes_position = tuple(position_expected.shape)
    for component in ("x", "y"):
        a = getattr(position_found, component)
        b = getattr(position_expected, component)
        a = na.broadcast_to(a, position_expected.shape).ndarray_aligned(axes_position)
        b = na.broadcast_to(b, position_expected.shape).ndarray_aligned(axes_position)
        assert np.allclose(a, b, rtol=0, atol=1e-9 * u.arcsec)
    timedelta = cast(na.ScalarArray, images.timedelta)
    index = {a: i for a, i in item.items() if a in timedelta.shape}
    assert np.all(found.timedelta == timedelta[index])
