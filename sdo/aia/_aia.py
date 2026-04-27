from typing import Literal
import astropy.units as u
import astropy.time
import named_arrays as na
from . import Filtergram

__all__ = [
    "open",
]


def open(
    time_start: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    wavelength: u.Quantity | na.ScalarArray = 304 * u.AA,
    user_email: None | str = None,
    series: Literal["aia.lev1_euv_12s", "aia.lev1_uv_24s"] = "aia.lev1_euv_12s",
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    limit: None | int = None,
) -> Filtergram:
    """
    Given a time range and a wavelength, download the corresponding
    AIA filtergram.

    .. important::

        Your email must be `registered with JSOC <http://jsoc.stanford.edu/ajax/register_email.html>`_
        to use this method.

    Parameters
    ----------
    time_start
        The start time of the search period
    time_stop
        The end time of the search period.
        If :obj:`None` (the default), this is set to one second after `time_start`
        which will have the effect of downloading one image.
    wavelength
        The wavelengths to download.
        Must be a valid AIA wavelength.
    user_email
        An email address used to notify the user that their JSOC request
        is complete.
        This email must be registered with JSOC before using this function.
        If :obj:`None`, the value is taken from the ``JSOC_EMAIL``
        environment variable.
    series
        The data series to download.
        See the `sunpy documentation <https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html#querying-the-jsoc>`_
        for more information.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    limit
        The maximum number of files to download for each wavelength.
    """

    if time_stop is None:
        time_start = astropy.time.Time(time_start)
        time_stop = time_start + 10 * u.s

    return Filtergram.from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        user_email=user_email,
        series=series,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        limit=limit,
    )
