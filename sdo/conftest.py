"""Refuse to contact JSOC from the tests where they may not, and stand in for it."""

from typing import Callable
import os
import socket
import pytest
import requests
import sunpy.net.jsoc
from sdo import _retry

jsoc = os.environ.get("SDO_TEST_JSOC", "1") != "0"
"""
Whether the tests may contact JSOC.

The test jobs of CI set the environment variable ``SDO_TEST_JSOC`` to ``0``,
since they open only what the download job found and downloaded from JSOC
and shared with them. A test which contacted JSOC there anyway would
otherwise pass, slowly, whenever JSOC answered, and the search or download
it should have found in what was shared would go unnoticed.
"""


def refuse_jsoc(getaddrinfo: Callable) -> Callable:
    """
    Wrap :func:`socket.getaddrinfo` so that it raises for any host of JSOC,
    and looks up any other as before.

    It raises a :class:`RuntimeError`, which neither :mod:`requests` nor
    :mod:`urllib` take for an error of the connection, so that the request
    is not made again, and the test fails at once.

    Parameters
    ----------
    getaddrinfo
        The function to wrap.
    """

    def wrapper(host: object, *args: object, **kwargs: object) -> object:
        """Look up a host, unless it is one of JSOC."""
        if isinstance(host, str) and host.endswith("stanford.edu"):
            raise RuntimeError(f"{host} was contacted, but SDO_TEST_JSOC=0")
        return getaddrinfo(host, *args, **kwargs)

    return wrapper


if not jsoc:
    # Here rather than in a fixture, since the tests download what they
    # open as they are collected, before any fixture runs.
    socket.getaddrinfo = refuse_jsoc(socket.getaddrinfo)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Stand in for the waits between retries, and return how long each was."""
    result = []
    monkeypatch.setattr(_retry, "_sleep", result.append)
    return result


path = "/SUM1/D1/S00000/image_lev1.fits"
"""The path of an image as JSOC serves it, which the tests stand in for."""

url = "http://jsoc.stanford.edu" + path
"""The URL of :data:`path`."""


def response(status: int, content: bytes = b"") -> requests.Response:
    """
    An answer of the server to :data:`url`, with a status and contents, as
    :mod:`requests` makes one.

    Parameters
    ----------
    status
        The status of the answer.
    content
        The contents of the answer.
    """
    result = requests.Response()
    result.status_code = status
    result._content = content
    result.url = url
    return result


@pytest.fixture
def jsoc_searches(monkeypatch: pytest.MonkeyPatch) -> Callable[[list], list]:
    """
    Stand in for the searches of JSOC.

    Returns a function which takes what each search in turn finds, or the
    error it raises, the last of them for every search after, and which
    returns the list of the searches made.
    """

    def stand_in(answers: list) -> list:
        """Answer the searches of JSOC with `answers`, in turn."""
        calls = []

        def search(
            self: sunpy.net.jsoc.JSOCClient,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Answer with the next of `answers`."""
            calls.append(args)
            answer = answers[min(len(calls), len(answers)) - 1]
            if isinstance(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr(sunpy.net.jsoc.JSOCClient, "search", search)
        return calls

    return stand_in


def pytest_configure(config: pytest.Config) -> None:
    """Register the mark of a test which contacts JSOC."""
    config.addinivalue_line(
        "markers",
        "jsoc: the test contacts JSOC, so it is skipped if SDO_TEST_JSOC=0",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip the tests which contact JSOC, if they may not."""
    if jsoc:
        return
    skip = pytest.mark.skip(reason="SDO_TEST_JSOC=0")
    for item in items:
        if "jsoc" in item.keywords:
            item.add_marker(skip)
