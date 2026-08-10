import pytest
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
