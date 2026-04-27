import pytest
import astropy.units as u
import named_arrays as na
import sdo


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        sdo.aia.open("2021-09-23T06:00"),
        sdo.aia.open(
            time_start="2021-09-23T06:00",
            wavelength=na.ScalarArray([304] * u.AA, "wavelength"),
            limit=1
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
