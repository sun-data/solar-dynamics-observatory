from typing import Callable
import pathlib
import urllib.error
import pytest
import astropy.table
import requests
import sunpy.net.attrs
import named_arrays as na
import sdo
from sdo._data import Unavailable, _search_jsoc
from sdo.conftest import path, response, url


def _urls(*urls: str) -> na.ScalarArray:
    """URLs along the axis of time, as :func:`sdo.download` takes them."""
    return na.stack(list(urls), axis="time").astype(object)


def _server(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[requests.Response],
) -> list:
    """
    Stand in for the server of the files of JSOC, which answers each download
    with each of `answers` in turn, and return the list of the downloads made.
    """
    calls = []

    def get(url: str, timeout: float) -> requests.Response:
        """Answer with the next of `answers`."""
        calls.append(url)
        return answers[len(calls) - 1]

    monkeypatch.setattr(requests, "get", get)
    return calls


def test_download_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """A download the server fails with a 503 is made again, and saved."""
    calls = _server(monkeypatch, [response(503), response(200, b"image")])

    result = sdo.download(_urls(url), cache=str(tmp_path))

    assert pathlib.Path(result.ndarray.item(0)).read_bytes() == b"image"
    assert calls == [url, url]
    assert sleeps == [10]


@pytest.mark.parametrize(
    argnames="url_bad",
    argvalues=[
        # what a search made while the storage of JSOC was down once gave
        "http://jsoc.stanford.eduNoDataDirectory",
        "http://jsoc.stanford.edu/",
        "http://jsoc.stanford.edu/SUM1/D1/S00000/",
    ],
)
def test_download_no_path(tmp_path: pathlib.Path, url_bad: str) -> None:
    """
    A URL without the path of a file is refused, rather than taken for the
    directory it would be downloaded into.
    """
    with pytest.raises(ValueError, match="no path of a file"):
        sdo.download(_urls(url_bad), cache=str(tmp_path))


def test_download_cut_short(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A download whose writing is cut short leaves no file, so that it is
    downloaded again the next time, rather than taken for the whole file.
    """
    _server(monkeypatch, [response(200, b"image")] * 2)

    write_bytes = pathlib.Path.write_bytes

    def write_bytes_cut_short(self: pathlib.Path, data: bytes) -> int:
        """Write the start of the file, and fail, as a full disk would."""
        write_bytes(self, data[:2])
        raise OSError("no space left on device")

    with monkeypatch.context() as context:
        context.setattr(pathlib.Path, "write_bytes", write_bytes_cut_short)
        with pytest.raises(OSError, match="no space left"):
            sdo.download(_urls(url), cache=str(tmp_path))

    assert not (tmp_path / path.lstrip("/")).exists()
    assert not list(tmp_path.rglob("*.part"))

    result = sdo.download(_urls(url), cache=str(tmp_path))
    assert pathlib.Path(result.ndarray.item(0)).read_bytes() == b"image"


def test_download_part_of_its_own(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A download writes into a file of a name of its own, so that it does not
    write into that of another process downloading the same file.
    """
    _server(monkeypatch, [response(200, b"image")])

    file = tmp_path / path.lstrip("/")
    file.parent.mkdir(parents=True)
    other = file.with_name(file.name + ".part")
    other.write_bytes(b"another download")

    sdo.download(_urls(url), cache=str(tmp_path))

    assert file.read_bytes() == b"image"
    assert other.read_bytes() == b"another download"


def _search() -> astropy.table.Table:
    """Search for the images of AIA, with :func:`sdo._data._search_jsoc`."""
    return _search_jsoc(
        sunpy.net.attrs.jsoc.Series("aia.lev1_euv_12s"),
        segment="image",
    )


def test_search_jsoc_retry(
    jsoc_searches: Callable[[list], list],
    sleeps: list[float],
) -> None:
    """A search which times out is made again."""
    timeout = urllib.error.URLError(TimeoutError("timed out"))
    calls = jsoc_searches([timeout, astropy.table.Table({"image": [path]})])

    assert list(_search()["image"]) == [path]
    assert len(calls) == 2
    assert sleeps == [10]


def test_search_jsoc_unavailable(
    jsoc_searches: Callable[[list], list],
    sleeps: list[float],
) -> None:
    """
    A search JSOC answers with ``"NoDataDirectory"`` in place of the paths of
    the files, as it does while its storage is down, raises
    :class:`sdo._data.Unavailable`, an :class:`OSError`, at once.
    """
    calls = jsoc_searches([astropy.table.Table({"image": ["NoDataDirectory"]})])

    with pytest.raises(OSError, match="answered 'NoDataDirectory'") as error:
        _search()

    assert isinstance(error.value, Unavailable)
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    argnames="answer",
    argvalues=[
        astropy.table.Table({"image": ["NoDataFile"]}),
        astropy.table.Table({"image": [""]}),
        astropy.table.Table([astropy.table.MaskedColumn([path], "image", mask=[True])]),
        astropy.table.Table({"other": [path]}),
    ],
)
def test_search_jsoc_no_path(
    jsoc_searches: Callable[[list], list],
    answer: astropy.table.Table,
) -> None:
    """
    A search JSOC answers without the path of the file of a record, other
    than while its storage is down, raises a :class:`ValueError`.
    """
    jsoc_searches([answer])

    with pytest.raises(ValueError, match="path for the `image`|of the `image`"):
        _search()
