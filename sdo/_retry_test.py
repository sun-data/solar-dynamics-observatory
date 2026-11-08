import email.message
import pathlib
import time
import urllib.error
import pytest
import numpy as np
import requests
import named_arrays as na
import sdo
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


_transient = [
    requests.ConnectTimeout("connect timed out"),
    requests.ConnectionError("connection refused"),
    requests.ReadTimeout("read timed out"),
    requests.HTTPError(response=_response(503)),
    urllib.error.URLError("timed out"),
    urllib.error.HTTPError(_url, 502, "Bad Gateway", email.message.Message(), None),
    TimeoutError("timed out"),
    ConnectionResetError("connection reset"),
]
"""Errors of requests to JSOC which may pass if the request is made again."""

_permanent = [
    requests.HTTPError(response=_response(404)),
    requests.HTTPError("no response"),
    urllib.error.HTTPError(_url, 404, "Not Found", email.message.Message(), None),
    ValueError("not an error of the network"),
]
"""Errors of requests to JSOC which will not pass."""


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Stand in for the waits between retries, and return how long each was."""
    result = []
    monkeypatch.setattr(time, "sleep", result.append)
    return result


@pytest.mark.parametrize(argnames="error", argvalues=_transient)
def test_retry(error: Exception, sleeps: list[float]) -> None:
    """A request which fails for a reason that may pass is made again until it succeeds."""
    calls = []

    def request() -> str:
        """Fail every time but the last."""
        calls.append(None)
        if len(calls) < _retry.attempts:
            raise error
        return "result"

    assert _retry.retry(request) == "result"
    assert len(calls) == _retry.attempts
    assert sleeps == [_retry.delay * 2**i for i in range(_retry.attempts - 1)]


@pytest.mark.parametrize(argnames="error", argvalues=_transient)
def test_retry_gives_up(error: Exception, sleeps: list[float]) -> None:
    """The error of a request which keeps failing is raised after the last attempt."""
    calls = []

    def request() -> str:
        """Fail every time."""
        calls.append(None)
        raise error

    with pytest.raises(type(error)):
        _retry.retry(request)
    assert len(calls) == _retry.attempts
    assert len(sleeps) == _retry.attempts - 1


@pytest.mark.parametrize(argnames="error", argvalues=_permanent)
def test_retry_permanent(error: Exception, sleeps: list[float]) -> None:
    """A request which fails for a reason that will not pass is not made again."""
    calls = []

    def request() -> str:
        """Fail every time."""
        calls.append(None)
        raise error

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

    urls = na.ScalarArray(np.array([_url], dtype=object), axes="time")
    result = sdo.download(urls, cache=str(tmp_path))

    path = pathlib.Path(result.ndarray.item(0))
    assert path.read_bytes() == b"image"
    assert calls == [_url, _url]
    assert sleeps == [_retry.delay]
