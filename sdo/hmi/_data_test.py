import pytest
import pathlib
import joblib
import named_arrays as na
import sdo

_time_start = "2021-09-23T06:00:00"
_time_stop = "2021-09-23T06:01:00"


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
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_search(
    time_start: str,
    time_stop: str,
    cache: None | str | joblib.Memory,
):
    result = sdo.hmi.search(
        time_start=time_start,
        time_stop=time_stop,
        cache=cache,
    )

    for key in result:
        assert isinstance(result[key], na.AbstractScalarArray)
        assert result[key].shape == result["url"].shape

    for i in result["url"].ndindex():
        assert isinstance(result["url"][i].ndarray, str)

    # The keywords which place an image on the sky, since HMI files do not
    # carry them and this is the only chance to collect them.
    for key in ("DATE-OBS", "CRVAL1", "CRPIX1", "CDELT1", "CROTA2"):
        assert key in result


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
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_urls(
    time_start: str,
    time_stop: str,
    cache: None | str | joblib.Memory,
):
    result = sdo.hmi.urls(
        time_start=time_start,
        time_stop=time_stop,
        cache=cache,
    )

    for i in result.ndindex():
        assert isinstance(result[i].ndarray, str)


@pytest.mark.parametrize(
    argnames="cache",
    argvalues=[
        sdo.directory_default,
    ],
)
def test_download(
    cache: None | str | joblib.Memory,
):
    urls = sdo.hmi.urls(
        time_start=_time_start,
        time_stop=_time_stop,
        cache=cache,
    )

    result = sdo.download(urls, cache=cache)

    for i in result.ndindex():
        path = pathlib.Path(result[i].ndarray)
        assert path.is_file()
