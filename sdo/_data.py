import joblib
import functools
import pathlib
import requests
import sunpy.net.attr
import sunpy.net.jsoc
import named_arrays as na
import sdo
from ._retry import Unavailable, retry

__all__ = [
    "download",
]


def download(
    urls: na.AbstractScalarArray,
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
    cache: None | str | joblib.Memory = sdo.memory,
) -> na.ScalarArray:
    """
    Download the given URLs to a specified directory.
    If `overwrite` is :obj:`False`, the file will not be downloaded if it exists.

    JSOC drops connections now and then, so a download which fails for a
    reason that may pass is made again, up to three times in all, after waits
    of 10 and then 20 seconds: a connection which cannot be made, or which is
    dropped, or a server too busy to answer (a status of 408, 429, 500, 502,
    503, or 504). A server which is slow to answer is not asked again, so as
    not to add to its load, nor is one which refuses the request, such as
    with a 404.

    This knows nothing about which instrument the URLs came from, so it is
    shared by every instrument in this package.

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
        # `joblib.Memory` keeps `location` as it was given, so a cache asked
        # for as a string leaves a string here, and `_download` wants
        # somewhere it can call `mkdir` on.
        location = cache.location
        directory = pathlib.Path(location) if location else sdo.directory_default

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

        if not components:
            raise ValueError(f"{url!r} has no path of a file to download")

        file = "/".join(components)

        path = directory / file

        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite or not path.exists():
            content = retry(functools.partial(_get, url))
            # Written beside the file and then moved into place, so that a
            # download cut short leaves no file which would be taken for
            # the whole one the next time.
            part = path.with_name(path.name + ".part")
            part.write_bytes(content)
            part.replace(path)

        result[i] = str(path)

    return result


def _get(url: str) -> bytes:
    """Download the contents of a URL, raising an error if the server refuses."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _search_jsoc(
    *attrs: sunpy.net.attr.Attr,
    segment: str,
) -> sunpy.net.jsoc.JSOCResponse:
    """
    Search JSOC, and search again each time the search fails for a reason
    that may pass, with :func:`sdo._retry.retry`.

    While its storage is down, JSOC answers with ``"NoDataDirectory"`` in
    place of the path of each file, which is raised as
    :class:`sdo._retry.Unavailable`, so that the search is made again, and
    is not cached.

    Parameters
    ----------
    attrs
        The attributes to search by.
    segment
        The segment of the series, whose paths are checked.
    """

    def request() -> sunpy.net.jsoc.JSOCResponse:
        """Search JSOC, and check the paths of the files it found."""
        response = sunpy.net.jsoc.JSOCClient().search(*attrs)
        for row in response:
            path = row.get(segment)
            if not path.startswith("/"):
                raise Unavailable(
                    f"JSOC answered {path!r} in place of the path of the "
                    f"`{segment}` of a record, as it does while its storage "
                    f"is down"
                )
        return response

    return retry(request)
