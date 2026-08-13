import pytest
import astropy.time
import numpy as np
import astropy.units as u
import named_arrays as na
import sdo

_time_start = "2021-09-23T06:00"


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        sdo.hmi.open(_time_start),
        sdo.hmi.open(
            time_start=_time_start,
            time_stop="2021-09-23T06:01:30",
        ),
    ],
)
class TestMagnetogram:

    def test_axis_time(self, array: sdo.hmi.Magnetogram):
        assert isinstance(array.axis_time, str)

    def test_axis_detector_x(self, array: sdo.hmi.Magnetogram):
        assert isinstance(array.axis_detector_x, str)

    def test_axis_detector_y(self, array: sdo.hmi.Magnetogram):
        assert isinstance(array.axis_detector_y, str)

    def test_outputs(self, array: sdo.hmi.Magnetogram):
        outputs = array.outputs
        assert na.unit(outputs).is_equivalent(u.G)

        # Off the disk there is no field to report, so a magnetogram is
        # expected to be part measurement and part nothing.
        finite = np.isfinite(outputs.ndarray.value)
        assert finite.any()
        assert not finite.all()

    def test_inputs(self, array: sdo.hmi.Magnetogram):
        inputs = array.inputs
        axis_time = array.axis_time

        assert inputs.time.shape[axis_time] == array.outputs.shape[axis_time]

        assert na.unit(inputs.crval.position.x).is_equivalent(u.arcsec)
        assert na.unit(inputs.cdelt.position.x).is_equivalent(u.arcsec)

    def test_position(self, array: sdo.hmi.Magnetogram):
        position = array.inputs[{array.axis_time: 0}].position

        # The Sun is about sixteen arcminutes across and HMI sees all of it,
        # so a field of view which does not is one the WCS has got wrong.
        radius = 960 * u.arcsec
        assert position.x.min().ndarray < -radius
        assert position.x.max().ndarray > +radius
        assert position.y.min().ndarray < -radius
        assert position.y.max().ndarray > +radius


def test_open_one():
    """Only a start time is only one image, whichever side of it lands closest."""
    result = sdo.hmi.open(_time_start)
    assert result.outputs.shape[result.axis_time] == 1


def test_position_against_fits_convention():
    """
    The coordinates must be the ones the FITS keywords describe.

    `CRPIX` counts pixels from one and
    :class:`named_arrays.AbstractWcsVector` counts them from zero, so this is
    a place an off-by-one can hide: it moves every image by a single pixel,
    which is far too small to notice in a picture and far too large to want.
    """
    time_start = astropy.time.Time(_time_start)

    found = sdo.hmi.search(
        time_start=time_start,
        time_stop=time_start + 45 * u.s,
    )
    array = sdo.hmi.open(_time_start)

    # The record `open` chose, which is the one nearest the time asked for.
    time_found = astropy.time.Time(np.ravel(found["DATE-OBS"].ndarray))
    time_array = astropy.time.Time(np.ravel(array.inputs.time.ndarray))[0]
    index = {"time": int(np.argmin(np.abs((time_found - time_array).to_value(u.s))))}

    def keyword(name):
        return float(found[name][index].ndarray)

    angle = keyword("CROTA2") * u.deg
    cos = float(np.cos(angle))
    sin = float(np.sin(angle))

    position = array.inputs[{array.axis_time: 0}].position

    # A few vertices, away from the middle so that a rotation matters.
    for index_x, index_y in ((0, 0), (1000, 3000), (4096, 4096)):

        # The FITS convention, written out: pixels counted from one, and the
        # vertex of index `j` lying half a pixel below the center of pixel
        # `j`.
        pixel_x = index_x + 0.5 - keyword("CRPIX1")
        pixel_y = index_y + 0.5 - keyword("CRPIX2")

        x = keyword("CDELT1") * (cos * pixel_x - sin * pixel_y)
        y = keyword("CDELT2") * (sin * pixel_x + cos * pixel_y)

        index_vertex = {
            array.axis_detector_x: index_x,
            array.axis_detector_y: index_y,
        }
        result = position[index_vertex]

        assert np.isclose(result.x.ndarray.to_value(u.arcsec), x, atol=1e-6)
        assert np.isclose(result.y.ndarray.to_value(u.arcsec), y, atol=1e-6)
