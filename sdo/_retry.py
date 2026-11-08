"""Make the requests to JSOC again when they fail for a reason that may pass."""

from typing import Callable, TypeVar
import http.client
import json
import logging
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

statuses = frozenset({408, 429, 500, 502, 503, 504})
"""
The statuses of an answer of the server which may pass: it timed out
waiting for the request, was asked too often, failed, or was too busy to
answer.
"""

_logger = logging.getLogger(__name__)
"""Says each time a request is made again, and why."""

_sleep = time.sleep
"""
Waits between the attempts. The tests stand in for this, and not for
:func:`time.sleep`, which everything else waits with.
"""


def transient(error: BaseException) -> bool:
    """
    Whether an error of a request to JSOC may pass if the request is made
    again.

    A connection which could not be made, or which was dropped before the
    whole answer came, may pass, as may an answer of the server in
    :data:`statuses`. A request the server answered too slowly is not made
    again, so as not to add to the load of a server which is already slow,
    and neither is one it refused, or any other error.

    Parameters
    ----------
    error
        The error the request raised. :mod:`requests` raises the errors of
        downloads, and :mod:`urllib`, through :mod:`drms`, those of searches.
    """
    if isinstance(error, requests.exceptions.SSLError):
        return False
    if isinstance(error, requests.HTTPError):
        response = error.response
        return response is not None and response.status_code in statuses
    if isinstance(error, urllib.error.HTTPError):
        return error.code in statuses
    if isinstance(error, urllib.error.URLError):
        # urllib raises this for a connection which could not be made, and
        # raises the errors of reading the answer as they are
        return isinstance(error.reason, (TimeoutError, ConnectionError))
    return isinstance(
        error,
        (
            # requests raises these for a connection which could not be made
            # or was dropped, and a `ReadTimeout`, which is not one, for an
            # answer too slow
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            # urllib raises these for an answer cut short
            ConnectionError,
            http.client.IncompleteRead,
            json.JSONDecodeError,
        ),
    )


def retry(request: Callable[[], _T]) -> _T:
    """
    Make a request to JSOC, and make it again after a wait each time it fails
    for a reason that may pass, up to :data:`attempts` times in all.

    JSOC drops connections from some places now and then, such as the
    runners of GitHub Actions, and answers with a 503 when it is busy. Each
    retry is logged as a warning.

    Parameters
    ----------
    request
        The request, which takes no arguments and returns its result.
    """
    for attempt in range(1, attempts + 1):
        try:
            return request()
        except Exception as error:
            if attempt == attempts or not transient(error):
                raise
            wait = delay * 2 ** (attempt - 1)
            _logger.warning(
                "A request to JSOC failed on attempt %d of %d, "
                "and will be made again in %d s: %r",
                attempt,
                attempts,
                wait,
                error,
            )
            _sleep(wait)
    raise ValueError(f"{attempts=} must be at least 1")  # pragma: nocover
