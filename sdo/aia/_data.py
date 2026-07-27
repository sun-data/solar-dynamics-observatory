import joblib
from typing import Literal
import pathlib
import requests
import astropy.units as u
import astropy.time
import astropy.io.fits
import sunpy.net.attrs
import sunpy.net.jsoc
import aiapy.calibrate.utils
import named_arrays as na
import sdo

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


def download(
    urls: na.AbstractScalarArray,
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
    cache: None | str | joblib.Memory = sdo.memory,
) -> na.ScalarArray:
    """
    Download the given URLs to a specified directory.
    If `overwrite` is :obj:`False`, the file will not be downloaded if it exists.

    Parameters
    ----------
    urls
        The URLs to download.
    directory
        The directory to place the downloaded files.
        If :obj:`None` (the default), the location of `cache` will be used.
    overwrite
        Boolean flag controlling whether to overwrite existing files.
    cache
        The location to cache the results of this function.
        If not provided, the default cache location, :attr:`sdo.memory` is used.
        If :obj:`None`, no caching is performed, and if `cache` is a pathlike,
        a new cache is created at that location.
    """

    if not isinstance(cache, joblib.Memory):
        cache = joblib.Memory(location=cache, verbose=False)

    if directory is None:
        directory = cache.location or sdo.directory_default

    return cache.cache(_download)(
        urls=urls,
        directory=directory,
        overwrite=overwrite,
    )


def _download(
    urls: na.AbstractScalarArray,
    directory: pathlib.Path,
    overwrite: bool = False,
) -> na.ScalarArray:
    directory.mkdir(parents=True, exist_ok=True)

    result = urls.copy()

    for i in urls.ndindex():
        url = urls[i].ndarray

        components = url.split("/")[3:]

        file = "/".join(components)

        path = directory / file

        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite or not path.exists():
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

        result[i] = str(path)

    return result


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


def _prep(
    files: na.AbstractScalarArray,
    register: bool = False,
) -> na.ScalarArray:
    result = files.copy()

    # Determine the time range spanned by all files so the pointing table only
    # needs to be fetched from JSOC once instead of once per file.
    times = []
    for i in files.ndindex():
        file = files[i].ndarray
        with astropy.io.fits.open(file) as hdul:
            for hdu in hdul:
                if "DATE-OBS" in hdu.header:
                    times.append(astropy.time.Time(hdu.header["DATE-OBS"]))
                    break
    time_min = min(times)
    time_max = max(times)

    pointing_table = aiapy.calibrate.utils.get_pointing_table(
        source="JSOC",
        time_range=(time_min - 12 * u.h, time_max + 12 * u.h),
    )

    for i in files.ndindex():
        file = pathlib.Path(files[i].ndarray)

        stem = file.stem + "5"
        if register:
            stem = stem + "_registered"

        file_15 = file.parent / (stem + file.suffix)

        if not file_15.is_file():

            aia_map = sunpy.map.Map(file)

            aia_map = aiapy.calibrate.update_pointing(
                smap=aia_map,
                pointing_table=pointing_table,
            )
            if register:
                aia_map = aiapy.calibrate.register(aia_map)
            aia_map.save(file_15)

        result[i] = str(file_15)

    return result
