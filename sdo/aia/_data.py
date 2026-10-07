import joblib
from typing import Literal
import functools
import pathlib
import astropy.units as u
import astropy.time
import astropy.table
import sunpy.net.attrs
import sunpy.net.jsoc
import aiapy.calibrate.utils
import named_arrays as na
import sdo
from .._data import download

__all__ = [
    "urls",
    "download",
    "prep",
]


def urls(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    wavelength: u.Quantity | na.ScalarArray,
    series: Literal["aia.lev1_euv_12s", "aia.lev1_uv_24s"] = "aia.lev1_euv_12s",
    axis_time: str = "time",
    limit: None | int = None,
    cache: None | str | joblib.Memory = sdo.memory,
) -> na.ScalarArray:
    """
    Given a time range and an array of wavelengths,
    find the URLs of the corresponding AIA observations.

    Parameters
    ----------
    time_start
        The start time of the search period
    time_stop
        The end time of the search period.
    wavelength
        The wavelengths to download.
        Must be a valid AIA wavelength.
    series
        The data series to download.
        See the `sunpy documentation <https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html#querying-the-jsoc>`_
        for more information.
    axis_time
        The logical axis corresponding to changes in time.
    limit
        The maximum number of files to download for each wavelength.
    cache
        The location to cache the results of this function.
        If not provided, the default cache location, :attr:`sdo.memory` is used.
        If :obj:`None`, no caching is performed, and if `cache` is a pathlike,
        a new cache is created at that location.
    """

    if not isinstance(cache, joblib.Memory):
        cache = joblib.Memory(location=cache, verbose=False)

    return cache.cache(_urls)(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        series=series,
        axis_time=axis_time,
        limit=limit,
    )


def _urls(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    wavelength: u.Quantity | na.ScalarArray,
    series: Literal["aia.lev1_euv_12s", "aia.lev1_uv_24s"] = "aia.lev1_euv_12s",
    axis_time: str = "time",
    limit: None | int = None,
) -> na.ScalarArray:
    time_start = astropy.time.Time(time_start)
    time_stop = astropy.time.Time(time_stop)

    wavelength = na.as_named_array(wavelength)
    if wavelength.ndim == 0:
        axis_wavelength = "wavelength"
        wavelength = wavelength.add_axes(axis_wavelength)
    elif wavelength.ndim == 1:
        axis_wavelength = wavelength.axes[0]
    else:  # pragma: nocover
        raise ValueError(f"`wavelength` must be 0D or 1D, got {wavelength.shape=}")

    attrs = (
        sunpy.net.attrs.jsoc.Segment("image"),
        sunpy.net.attrs.jsoc.Series(series),
    )

    if limit is not None:
        timedelta = (time_stop - time_start).to(u.s)
        time_start = time_start + timedelta / 2
        period = timedelta / limit
        attrs = attrs + (sunpy.net.attrs.Sample(period),)

    attrs = attrs + (sunpy.net.attrs.Time(time_start, time_stop),)

    url_base = "http://jsoc.stanford.edu"

    urls = []

    client = sunpy.net.jsoc.JSOCClient()

    for w in wavelength.ndindex():
        channel = wavelength[w].ndarray

        attrs_w = attrs + (sunpy.net.attrs.jsoc.Wavelength(channel),)

        response = client.search(*attrs_w)

        urls_w = []

        for row in response:
            url = url_base + row.get("image")

            urls_w.append(url)

        urls_w = na.stack(urls_w, axis=axis_time)

        urls.append(urls_w)

    urls = na.stack(urls, axis=axis_wavelength)

    urls = urls.transpose((axis_time, axis_wavelength))

    return urls.astype(object)


def prep(
    files: na.AbstractScalarArray,
    register: bool = False,
    cache: None | str | joblib.Memory = sdo.memory,
) -> na.ScalarArray:
    """
    Convert an array of FITS files from Level 1 to Level 1.5 using :mod:`aiapy`.

    Parameters
    ----------
    files
        The array of Level 1 FITS files to convert.
    register
        Boolean flag controlling whether the images are also registered using
        :func:`aiapy.calibrate.register`, which rotates each image to solar
        north up and scales it to a common plate scale.
        Registered and unregistered results are saved to different files,
        so both can coexist in the same cache.
    cache
        The location to cache the results of this function.
        If not provided, the default cache location, :attr:`sdo.memory` is used.
        If :obj:`None`, no caching is performed, and if `cache` is a pathlike,
        a new cache is created at that location.
    """

    if not isinstance(cache, joblib.Memory):
        cache = joblib.Memory(location=cache, verbose=False)

    return cache.cache(_prep)(
        files=files,
        register=register,
    )


@functools.cache
def _pointing_table() -> astropy.table.QTable:
    """
    The copy of the JSOC pointing table LMSAL keeps for the whole mission,
    synced daily, so that preparing images does not depend on the JSOC
    answering a query in time.

    :mod:`aiapy` downloads all 22 MB of it on every call, so it is fetched
    once per process, and only when an image is prepared, and again only if
    an image is newer than its last entry.
    """
    return aiapy.calibrate.utils.get_pointing_table(source="lmsal")


def _prep(
    files: na.AbstractScalarArray,
    register: bool = False,
) -> na.ScalarArray:
    result = files.copy()

    for i in files.ndindex():
        file = pathlib.Path(files[i].ndarray)

        stem = file.stem + "5"
        if register:
            stem = stem + "_registered"

        file_15 = file.parent / (stem + file.suffix)

        if not file_15.is_file():

            aia_map = sunpy.map.Map(file)

            if aia_map.reference_date >= _pointing_table()["T_STOP"].max():
                # Fetched earlier in this process, so perhaps before the
                # daily sync which added this image.
                _pointing_table.cache_clear()

            aia_map = aiapy.calibrate.update_pointing(
                smap=aia_map,
                pointing_table=_pointing_table(),
            )
            if register:
                aia_map = aiapy.calibrate.register(aia_map)
            aia_map.save(file_15)

        result[i] = str(file_15)

    return result
