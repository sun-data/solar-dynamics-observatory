from typing import Callable
import email.message
import http.client
import json
import logging
import urllib.error
import pytest
import requests
import sdo.conftest
from sdo import _retry
from sdo.conftest import response, url


def _http_error(status: int) -> urllib.error.HTTPError:
    """An answer of the server with a status, as urllib raises it."""
    return urllib.error.HTTPError(url, status, "", email.message.Message(), None)


_transient = [
    requests.ConnectTimeout("connect timed out"),
    requests.ConnectionError("connection refused"),
    requests.exceptions.ChunkedEncodingError("connection dropped"),
    requests.HTTPError(response=response(503)),
    requests.HTTPError(response=response(429)),
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
    requests.HTTPError(response=response(404)),
    requests.HTTPError(response=response(501)),
    requests.HTTPError("no response"),
    urllib.error.URLError("unknown url type"),
    _http_error(404),
    TimeoutError("timed out"),
    ValueError("not an error of the network"),
]
"""Errors of requests to JSOC which will not pass, or should not be made again."""


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
