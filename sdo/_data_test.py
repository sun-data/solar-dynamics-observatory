import pathlib
import pytest
import astropy.table
import requests
import sunpy.net.attrs
import sunpy.net.jsoc
import named_arrays as na
import sdo
from sdo import _retry
from sdo._data import _search_jsoc

_path = "/SUM1/D1/S00000/image_lev1.fits"
"""The path of an image as JSOC serves it, which is never downloaded."""

_url = "http://jsoc.stanford.edu" + _path
"""The URL of :data:`_path`."""


def _searches(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[astropy.table.Table],
) -> list:
    """
    Stand in for searches of JSOC, which find each of `responses` in turn,
    and return the list of the searches made.
    """
    calls = []

    def search(
        self: sunpy.net.jsoc.JSOCClient,
        *args: object,
        **kwargs: object,
    ) -> astropy.table.Table:
        """Find the next of `responses`."""
        calls.append(args)
        return responses[len(calls) - 1]

    monkeypatch.setattr(sunpy.net.jsoc.JSOCClient, "search", search)
    return calls


_no_data_directory = astropy.table.Table({"image": ["NoDataDirectory"]})
"""The answer of JSOC to a search while its storage is down."""


def test_search_jsoc_no_data_directory(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """
    A search JSOC answers without the paths of the files, as it does while
    its storage is down, is made again.
    """
    found = astropy.table.Table({"image": [_path]})
    calls = _searches(monkeypatch, [_no_data_directory, found])

    response = _search_jsoc(
        sunpy.net.attrs.jsoc.Series("aia.lev1_euv_12s"),
        segment="image",
    )

    assert list(response["image"]) == [_path]
    assert len(calls) == 2
    assert sleeps == [10]


def test_search_jsoc_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """A search JSOC answers without the paths of the files every time says so."""
    calls = _searches(monkeypatch, [_no_data_directory] * 3)

    with pytest.raises(_retry.Unavailable, match="NoDataDirectory"):
        _search_jsoc(
            sunpy.net.attrs.jsoc.Series("aia.lev1_euv_12s"),
            segment="image",
        )

    assert len(calls) == 3
    assert sleeps == [10, 20]


def test_download_no_path(tmp_path: pathlib.Path) -> None:
    """
    A URL without the path of a file, as a search made while the storage of
    JSOC was down once gave, is refused, rather than taken for the directory
    it would be downloaded into.
    """
    urls = na.stack(["http://jsoc.stanford.eduNoDataDirectory"], axis="time")

    with pytest.raises(ValueError, match="no path of a file"):
        sdo.download(urls.astype(object), cache=str(tmp_path))


def test_download_cut_short(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A download whose writing is cut short leaves no file, so that it is
    downloaded again the next time, rather than taken for the whole file.
    """

    def get(url: str, timeout: float) -> requests.Response:
        """Stand in for the server, which sends the image."""
        response = requests.Response()
        response.status_code = 200
        response._content = b"image"
        return response

    monkeypatch.setattr(requests, "get", get)

    write_bytes = pathlib.Path.write_bytes

    def write_bytes_cut_short(self: pathlib.Path, data: bytes) -> int:
        """Write the start of the file, and fail, as a full disk would."""
        write_bytes(self, data[:2])
        raise OSError("no space left on device")

    urls = na.stack([_url], axis="time").astype(object)

    with monkeypatch.context() as context:
        context.setattr(pathlib.Path, "write_bytes", write_bytes_cut_short)
        with pytest.raises(OSError, match="no space left"):
            sdo.download(urls, cache=str(tmp_path))

    assert not (tmp_path / _path.lstrip("/")).exists()

    result = sdo.download(urls, cache=str(tmp_path))
    assert pathlib.Path(result.ndarray.item(0)).read_bytes() == b"image"
