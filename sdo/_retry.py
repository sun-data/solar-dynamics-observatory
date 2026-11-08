"""Make the requests to JSOC again when they fail for a reason that may pass."""

from typing import Callable, TypeVar
import time
import urllib.error
import requests

__all__ = [
    "retry",
]

_T = TypeVar("_T")

attempts = 3
"""How many times in all a request to JSOC is made before its error is raised."""

delay = 10
"""
How many seconds to wait before the first retry, doubled before each one
after it.
"""


def transient(error: Exception) -> bool:
    """
    Whether an error of a request to JSOC may pass if the request is made
    again.

    A connection which failed or timed out, or an error of the server (a
    status of 500 or more), may pass. A request the server refused (a status
    from 400 to 499), or any other error, will not.

    Parameters
    ----------
    error
        The error the request raised. :mod:`requests` raises the errors of
        downloads, and :mod:`urllib`, through :mod:`drms`, those of searches.
    """
    if isinstance(error, requests.HTTPError):
        response = error.response
        return response is not None and response.status_code >= 500
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    return isinstance(
        error,
        (
            requests.ConnectionError,
            requests.Timeout,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ),
    )


def retry(request: Callable[[], _T]) -> _T:
    """
    Make a request to JSOC, and make it again after a wait each time it fails
    for a reason that may pass, up to :data:`attempts` times in all.

    JSOC drops connections from some places now and then, such as the
    runners of GitHub Actions, and answers with a 503 when it is busy.

    Parameters
    ----------
    request
        The request, which takes no arguments and returns its result.
    """
    for attempt in range(attempts):
        try:
            return request()
        except Exception as error:
            if attempt == attempts - 1 or not transient(error):
                raise
            time.sleep(delay * 2**attempt)
    raise ValueError(f"{attempts=} must be at least 1")  # pragma: nocover
