import joblib
from typing import Literal
import astropy.units as u
import astropy.time
import sunpy.net.attrs
import sunpy.net.jsoc
import named_arrays as na
import sdo

__all__ = [
    "search",
    "urls",
]

#: The keywords needed to place an image on the sky, which for HMI are not
#: written into the file and have to be asked for alongside it.
_keys_wcs = (
    "DATE-OBS",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "CDELT1",
    "CDELT2",
    "CROTA2",
    "CUNIT1",
    "CUNIT2",
    "BUNIT",
)


def _cadence(series: str) -> u.Quantity:
    """
    How long one image of a series stands for.

    The series is named for its cadence, so the name is where this comes
    from.

    Parameters
    ----------
    series
        The data series, named for its cadence.
    """
    suffix = series.rsplit("_", 1)[-1]
    if suffix.endswith("s") and suffix[:-1].isdigit():
        return int(suffix[:-1]) * u.s
    else:  # pragma: nocover
        return 45 * u.s


def search(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    series: Literal["hmi.M_45s", "hmi.M_720s"] = "hmi.M_45s",
    segment: str = "magnetogram",
    axis_time: str = "time",
    limit: None | int = None,
    cache: None | str | joblib.Memory = sdo.memory,
) -> dict[str, na.ScalarArray]:
    """
    Find the HMI observations in a time range, and the keywords describing
    them.

    Where AIA writes its keywords into the file it serves, the files JSOC
    serves for HMI are the bare image: the primary header has six cards and
    the image none of the ones which would place it on the sky. Everything
    needed to do that is instead held as a keyword of the series, so it is
    asked for at the same time as the file and carried alongside it.

    Parameters
    ----------
    time_start
        The start time of the search period.
    time_stop
        The end time of the search period.
    series
        The data series to download.
        See the `sunpy documentation <https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html#querying-the-jsoc>`_
        for more information.
    segment
        The segment of the series to download, which is the quantity being
        asked for: ``magnetogram`` for the line-of-sight magnetic field,
        ``continuum`` for the continuum intensity, ``Dopplergram`` for the
        line-of-sight velocity.
    axis_time
        The logical axis corresponding to changes in time.
    limit
        The maximum number of files to download.
    cache
        The location to cache the results of this function.
        If not provided, the default cache location, :attr:`sdo.memory` is
        used. If :obj:`None`, no caching is performed, and if `cache` is a
        pathlike, a new cache is created at that location.
    """

    if not isinstance(cache, joblib.Memory):
        cache = joblib.Memory(location=cache, verbose=False)

    return cache.cache(_search)(
        time_start=time_start,
        time_stop=time_stop,
        series=series,
        segment=segment,
        axis_time=axis_time,
        limit=limit,
    )


def _search(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    series: str = "hmi.M_45s",
    segment: str = "magnetogram",
    axis_time: str = "time",
    limit: None | int = None,
) -> dict[str, na.ScalarArray]:
    time_start = astropy.time.Time(time_start)
    time_stop = astropy.time.Time(time_stop)

    attrs = (
        sunpy.net.attrs.jsoc.Segment(segment),
        sunpy.net.attrs.jsoc.Series(series),
    )

    if limit is not None:
        timedelta = (time_stop - time_start).to(u.s)
        time_start = time_start + timedelta / 2
        # No shorter than one cadence, since JSOC expresses a sampling
        # period as a whole number of slots and rounds to the nearest one:
        # asking for more images than the range holds rounds down to a step
        # of no slots at all, which it then divides by.
        period = max(timedelta / limit, _cadence(series))
        attrs = attrs + (sunpy.net.attrs.Sample(period),)

    attrs = attrs + (sunpy.net.attrs.Time(time_start, time_stop),)

    url_base = "http://jsoc.stanford.edu"

    client = sunpy.net.jsoc.JSOCClient()

    response = client.search(*attrs)

    # The column holding the path is named for the segment, where AIA calls
    # its one segment `image`.
    result = {
        "url": na.stack(
            [url_base + row.get(segment) for row in response],
            axis=axis_time,
        ).astype(object),
    }

    for key in _keys_wcs:
        if key not in response.colnames:  # pragma: nocover
            continue
        result[key] = na.stack(
            [row.get(key) for row in response],
            axis=axis_time,
        )

    return result


def urls(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    series: Literal["hmi.M_45s", "hmi.M_720s"] = "hmi.M_45s",
    segment: str = "magnetogram",
    axis_time: str = "time",
    limit: None | int = None,
    cache: None | str | joblib.Memory = sdo.memory,
) -> na.ScalarArray:
    """
    Given a time range, find the URLs of the corresponding HMI observations.

    This is :func:`search` without the keywords, for symmetry with
    :func:`sdo.aia.urls`.

    Parameters
    ----------
    time_start
        The start time of the search period.
    time_stop
        The end time of the search period.
    series
        The data series to download.
    segment
        The segment of the series to download.
    axis_time
        The logical axis corresponding to changes in time.
    limit
        The maximum number of files to download.
    cache
        The location to cache the results of this function.
    """
    return search(
        time_start=time_start,
        time_stop=time_stop,
        series=series,
        segment=segment,
        axis_time=axis_time,
        limit=limit,
        cache=cache,
    )["url"]
