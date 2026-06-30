import joblib
from typing import Literal
import os
import pathlib
import requests
import numpy as np
import astropy.units as u
import astropy.time
import sunpy.net.attrs
import sunpy.net.jsoc
import aiapy
import named_arrays as na
import sdo

__all__ = [
    "urls_jsoc",
    "download"
]


def urls_jsoc(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    wavelength: None | u.Quantity | na.ScalarArray,
    user_email: None | str = None,
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
    user_email
        An email address used to notify the user that their JSOC request
        is complete.
        This email must be registered with JSOC before using this function.
        If :obj:`None`, the value is taken from the ``JSOC_EMAIL``
        environment variable.
    series
        The data series to download.
        See the `sunpy documentation <https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html#querying-the-jsoc>`_
        for more information.
    axis_time
        The logical axis corresponding to changes in time.
    limit
        The maximum number of files to download for each wavelength.
    cache
        The location to cache the results of this function to avoid repeated
        queries to JSOC.
        If not provided, the default cache location, :attr:`sdo.memory` is used.
        If :obj:`None`, no caching is performed, and if `cache` is a pathlike,
        a new cache is created at that location.
    """

    if not isinstance(cache, joblib.Memory):
        cache = joblib.Memory(location=cache, verbose=False)

    return cache.cache(_urls_jsoc)(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        user_email=user_email,
        series=series,
        axis_time=axis_time,
        limit=limit,
    )


def _urls_jsoc(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    wavelength: None | u.Quantity | na.ScalarArray,
    user_email: None | str = None,
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

    if user_email is None:
        user_email = os.environ["JSOC_EMAIL"]

    attrs = (
        sunpy.net.attrs.jsoc.Notify(user_email),
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

    for w in wavelength.ndindex():
        channel = wavelength[w].ndarray

        attrs_w = attrs + (sunpy.net.attrs.jsoc.Wavelength(channel),)

        response = sunpy.net.jsoc.search(*attrs_w)

        urls_w = []

        for row in response:

            url = url_base + row[0].get("image")

            urls_w.append(url)

        urls_w = na.stack(urls_w, axis=axis_wavelength)

        urls.append(urls_w)

    urls = na.stack(urls, axis=axis_time)

    return urls


def download(
    urls: na.AbstractScalarArray,
    cache: None | str | joblib.Memory = sdo.memory,
) -> list[pathlib.Path]:

    if not isinstance(cache, joblib.Memory):
        cache = joblib.Memory(location=cache, verbose=False)

    return cache.cache(_download)(
        urls=urls,
        directory=cache.location,
    )


def _download(
    urls: na.AbstractScalarArray,
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
) -> list[pathlib.Path]:
    """
    Download the given URLs to a specified directory.
    If `overwrite` is :obj:`False`, the file will not be downloaded if it exists.

    Parameters
    ----------
    urls
        The URLs to download.
    directory
        The directory to place the downloaded files.
    overwrite
        Boolean flag controlling whether to overwrite existing files.

    """
    if directory is None:
        directory = sdo.directory_default

    directory.mkdir(parents=True, exist_ok=True)

    result = urls.copy()

    for i in urls.ndindex():

        url = urls[i]

        file = directory / url.split("/")[~0]

        if overwrite or not file.exists():
            r = requests.get(url, stream=True)
            with open(file, "wb") as f:
                f.write(r.content)

        result[i] = file

    return result
