import pytest
import pathlib
import joblib
import numpy as np
import astropy.time
import astropy.units as u
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
    argnames="limit",
    argvalues=[
        None,
        # Fewer images than asked for are in the range, which JSOC expresses
        # as a sampling period of no slots and then divides by.
        3,
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
    limit: None | int,
    cache: None | str | joblib.Memory,
):
    result = sdo.hmi.search(
        time_start=time_start,
        time_stop=time_stop,
        limit=limit,
        cache=cache,
    )

    if limit is not None:
        assert result["url"].size <= limit

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


def test_search_limit_spans_the_range():
    """
    A limited search must sample the whole range, not the end of it.

    Asking for a sampling period is not the same as asking for fewer images
    out of the second half, which is what shifting the start of the range to
    its middle would give.
    """
    time_start = astropy.time.Time("2021-09-23T06:00:00")
    time_stop = time_start + 1 * u.hour
    limit = 4

    found = sdo.hmi.search(
        time_start=time_start,
        time_stop=time_stop,
        limit=limit,
    )

    time = astropy.time.Time(np.ravel(found["DATE-OBS"].ndarray))

    assert time.size <= limit

    # The first image has to lie within the first of the `limit` parts the
    # range is divided into, or the beginning of it went unsampled.
    fraction = (time[0] - time_start) / (time_stop - time_start)
    assert fraction < 1 / limit


@pytest.mark.parametrize(
    argnames="limit",
    argvalues=[
        1,
        2,
    ],
)
def test_search_limit_is_a_maximum(limit: int):
    """
    Never more images than were asked for, however JSOC rounds the range.

    A range one cadence wide holds one image and comes back holding the two
    which straddle it, so this is a real question rather than a formality.
    """
    time_start = astropy.time.Time("2021-09-23T06:00:00")

    found = sdo.hmi.search(
        time_start=time_start,
        time_stop=time_start + 45 * u.s,
        limit=limit,
    )

    assert found["url"].size <= limit


def test_download_cache_str(tmp_path: pathlib.Path):
    """
    A cache asked for as a string is a place, and has to be usable as one.

    :class:`joblib.Memory` keeps `location` as it was handed over, so a
    string stays a string all the way to where a directory is made.
    """
    urls = sdo.hmi.urls(
        time_start=_time_start,
        time_stop=_time_stop,
    )

    result = sdo.download(urls, cache=str(tmp_path / "cache"))

    for i in result.ndindex():
        assert pathlib.Path(result[i].ndarray).is_file()
