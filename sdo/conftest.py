"""Refuse to contact JSOC from the tests, where they may not."""

from typing import Callable
import os
import socket
import pytest
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
