import pytest
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        sdo.aia.Filtergram.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T06:01"),
            wavelength=304 * u.AA,
        ),
        sdo.aia.Filtergram.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T06:01"),
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
