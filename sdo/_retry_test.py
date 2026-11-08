from typing import Callable
import email.message
import http.client
import json
import logging
import pathlib
import urllib.error
import pytest
import astropy.table
import astropy.units as u
import requests
import sunpy.net.jsoc
import named_arrays as na
import sdo
import sdo.conftest
from sdo import _retry

_url = "http://jsoc.stanford.edu/SUM1/D1/S00000/image_lev1.fits"
"""A URL of an image as JSOC serves it, which is never downloaded."""


def _response(status: int, content: bytes = b"") -> requests.Response:
    """A response of the server, with a status and contents, as requests makes one."""
    response = requests.Response()
    response.status_code = status
    response._content = content
    response.url = _url
    return response


def _http_error(status: int) -> urllib.error.HTTPError:
    """An answer of the server with a status, as urllib raises it."""
    return urllib.error.HTTPError(_url, status, "", email.message.Message(), None)


_transient = [
    requests.ConnectTimeout("connect timed out"),
    requests.ConnectionError("connection refused"),
    requests.exceptions.ChunkedEncodingError("connection dropped"),
    requests.HTTPError(response=_response(503)),
    requests.HTTPError(response=_response(429)),
    urllib.error.URLError(TimeoutError("timed out")),
    urllib.error.URLError(ConnectionRefusedError("connection refused")),
    _http_error(502),
    _http_error(408),
    http.client.RemoteDisconnected("connection dropped"),
    http.client.IncompleteRead(b"partial"),
    json.JSONDecodeError("cut short", '{"status": ', 11),
]
"""Errors of requests to JSOC which may pass if the request is made again."""

_permanent = [
    requests.ReadTimeout("read timed out"),
    requests.exceptions.SSLError("certificate verify failed"),
    requests.HTTPError(response=_response(404)),
    requests.HTTPError(response=_response(501)),
    requests.HTTPError("no response"),
    urllib.error.URLError("unknown url type"),
    _http_error(404),
    TimeoutError("timed out"),
    ValueError("not an error of the network"),
]
"""Errors of requests to JSOC which will not pass, or should not be made again."""


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Stand in for the waits between retries, and return how long each was."""
    result = []
    monkeypatch.setattr(_retry, "_sleep", result.append)
    return result


def _failing(
    error: Exception,
    failures: int,
) -> tuple[list, Callable[[], str]]:
    """
    A request which raises `error` the first `failures` times it is made, and
    then returns ``"result"``, and the list of the times it was made.
    """
    calls = []

    def request() -> str:
        """Fail the first `failures` times."""
        calls.append(None)
        if len(calls) <= failures:
            raise error
        return "result"

    return calls, request


@pytest.mark.parametrize(argnames="error", argvalues=_transient)
def test_retry(
    error: Exception,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A request which fails for a reason that may pass is made again until it succeeds."""
    calls, request = _failing(error, failures=2)

    with caplog.at_level(logging.WARNING, logger="sdo._retry"):
        assert _retry.retry(request) == "result"

    assert len(calls) == 3
    assert sleeps == [10, 20]
    assert len(caplog.records) == 2
    assert repr(error) in caplog.records[0].getMessage()


@pytest.mark.parametrize(argnames="error", argvalues=_transient)
def test_retry_gives_up(error: Exception, sleeps: list[float]) -> None:
    """The error of a request which keeps failing is raised after the last attempt."""
    calls, request = _failing(error, failures=3)

    with pytest.raises(type(error)):
        _retry.retry(request)
    assert len(calls) == 3
    assert sleeps == [10, 20]


@pytest.mark.parametrize(argnames="error", argvalues=_permanent)
def test_retry_permanent(error: Exception, sleeps: list[float]) -> None:
    """A request which fails for a reason that will not pass is not made again."""
    calls, request = _failing(error, failures=1)

    with pytest.raises(type(error)):
        _retry.retry(request)
    assert len(calls) == 1
    assert sleeps == []


def test_download_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """A download the server fails with a 503 is made again, and saved."""
    responses = [_response(503), _response(200, b"image")]
    calls = []

    def get(url: str, timeout: float) -> requests.Response:
        """Stand in for the server, busy the first time and not the second."""
        calls.append(url)
        return responses[len(calls) - 1]

    monkeypatch.setattr(requests, "get", get)

    urls = na.stack([_url], axis="time").astype(object)
    result = sdo.download(urls, cache=str(tmp_path))

    path = pathlib.Path(result.ndarray.item(0))
    assert path.read_bytes() == b"image"
    assert calls == [_url, _url]
    assert sleeps == [10]


def _search_failing_once(
    monkeypatch: pytest.MonkeyPatch,
    response: astropy.table.Table,
) -> list:
    """
    Stand in for searches of JSOC, the first of which times out, and the
    rest of which find `response`, and return the list of the searches made.
    """
    calls = []

    def search(
        self: sunpy.net.jsoc.JSOCClient,
        *args: object,
        **kwargs: object,
    ) -> astropy.table.Table:
        """Time out the first time, and find `response` after."""
        calls.append(args)
        if len(calls) == 1:
            raise urllib.error.URLError(TimeoutError("timed out"))
        return response

    monkeypatch.setattr(sunpy.net.jsoc.JSOCClient, "search", search)
    return calls


def test_aia_urls_retry(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """An AIA search which times out is made again."""
    path = "/SUM1/D1/S00000/image_lev1.fits"
    calls = _search_failing_once(monkeypatch, astropy.table.Table({"image": [path]}))

    result = sdo.aia.urls(
        time_start="2021-09-23T06:00:00",
        time_stop="2021-09-23T06:00:12",
        wavelength=171 * u.AA,
        cache=None,
    )

    assert result.ndarray.item(0) == "http://jsoc.stanford.edu" + path
    assert len(calls) == 2
    assert sleeps == [10]


def test_hmi_search_retry(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    """An HMI search which times out is made again."""
    path = "/SUM1/D1/S00000/magnetogram.fits"
    response = astropy.table.Table({"magnetogram": [path]})
    calls = _search_failing_once(monkeypatch, response)

    result = sdo.hmi.search(
        time_start="2021-09-23T06:00:00",
        time_stop="2021-09-23T06:01:00",
        cache=None,
    )

    assert result["url"].ndarray.item(0) == "http://jsoc.stanford.edu" + path
    assert len(calls) == 2
    assert sleeps == [10]


def test_refuse_jsoc() -> None:
    """The tests refuse a host of JSOC when they may not contact it, and no other."""
    calls = []

    def getaddrinfo(host: str, *args: object, **kwargs: object) -> list:
        """Stand in for looking up a host."""
        calls.append(host)
        return []

    refuse = sdo.conftest.refuse_jsoc(getaddrinfo)

    with pytest.raises(RuntimeError, match="jsoc1.stanford.edu"):
        refuse("jsoc1.stanford.edu", 443)
    assert refuse("hesperia.gsfc.nasa.gov", 443) == []
    assert calls == ["hesperia.gsfc.nasa.gov"]
