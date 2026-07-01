import pytest
import pathlib
import joblib
import astropy.units as u
import named_arrays as na
import sdo

_time_start = "2021-09-23T06:00:00"
_time_stop = "2021-09-23T06:00:12"

_wavelength = 304 * u.AA

_urls = sdo.aia.urls(
    time_start="2021-09-23T06:00",
    time_stop=_time_stop,
    wavelength=_wavelength,
)

_files = sdo.aia.download(_urls)


@pytest.mark.parametrize(
    argnames="time_start",
    argvalues=[
        _time_start,
    ],
)
@pytest.mark.parametrize(
    argnames="time_stop",
    argvalues=[
        _time_stop,
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        _wavelength,
    ],
)
@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_urls(
    time_start: str,
    time_stop: str,
    wavelength: u.Quantity | na.ScalarArray,
    cache: None | str | joblib.Memory,
):
    result = sdo.aia.urls(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        cache=cache,
    )

    for i in result.ndindex():
        assert isinstance(result[i].ndarray, str)


@pytest.mark.parametrize(
    argnames="urls",
    argvalues=[
        _urls,
    ],
)
@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_download(
    urls: na.ScalarArray,
    cache: None | str | joblib.Memory,
):
    result = sdo.aia.download(urls, cache=cache)

    for i in result.ndindex():
        path = pathlib.Path(result[i].ndarray)
        assert path.is_file()


@pytest.mark.parametrize(
    argnames="files",
    argvalues=[
        _files,
    ],
)
@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_prep(
    files: na.ScalarArray,
    cache: None | str | joblib.Memory,
):
    result = sdo.aia.prep(
        files=files,
        cache=cache,
    )

    for i in result.ndindex():
        path = pathlib.Path(result[i].ndarray)
        assert path.is_file()
