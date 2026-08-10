import joblib
import pathlib
import requests
import named_arrays as na
import sdo

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
