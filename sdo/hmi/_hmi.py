from typing import Literal
import joblib
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import sdo
from . import Magnetogram
from ._data import _cadence

__all__ = [
    "open",
]


def open(
    time_start: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    series: Literal["hmi.M_45s", "hmi.M_720s"] = "hmi.M_45s",
    segment: str = "magnetogram",
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    limit: None | int = None,
    cache: None | str | joblib.Memory = sdo.memory,
) -> Magnetogram:
    """
    Given a time range, download the corresponding HMI images.

    Parameters
    ----------
    time_start
        The start time of the search period.
    time_stop
        The end time of the search period.
        If :obj:`None` (the default), the single image closest to
        `time_start` is downloaded.
    series
        The data series to download.
        See the `sunpy documentation <https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html#querying-the-jsoc>`_
        for more information.
    segment
        The segment of the series to download, which is the quantity being
        asked for: ``magnetogram`` for the line-of-sight magnetic field,
        ``continuum`` for the continuum intensity, ``Dopplergram`` for the
        line-of-sight velocity.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector
        :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector
        :math:`y`-coordinate.
    limit
        The maximum number of files to download.
        Ignored when `time_stop` is :obj:`None`, since exactly one image is
        downloaded then whatever this says.
    cache
        The location to cache the results of this function.
        If not provided, the default cache location, :attr:`sdo.memory` is
        used. If :obj:`None`, no caching is performed, and if `cache` is a
        pathlike, a new cache is created at that location.

    Examples
    --------

    Download one magnetogram and show it.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        import sdo

        a = sdo.hmi.open("2019-09-30 18:08:00")

        b = a.outputs[{a.axis_time: 0}]

        # Saturated well below the strongest fields, since otherwise the
        # active regions are the only thing with any contrast at all.
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.imshow(
            b.value,
            axis_x=a.axis_detector_x,
            axis_y=a.axis_detector_y,
            ax=ax,
            cmap="gray",
            vmin=-100,
            vmax=+100,
        );
    """

    if time_stop is not None:
        return Magnetogram.from_time_range(
            time_start=time_start,
            time_stop=time_stop,
            series=series,
            segment=segment,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
            limit=limit,
            cache=cache,
        )

    # Only a start time was given, so exactly one image is wanted. A window
    # one cadence wide is sure to contain one, and is not sure to contain
    # only one: JSOC matches on `T_REC`, which is a slot in TAI rather than
    # the UTC time the image is stamped with, so a window can catch the
    # record on either side of it as well. Narrowing the window does not
    # help, since it is the offset and not the width which decides this.
    # Which record was asked for is therefore settled here rather than by
    # JSOC, and the ones which were not asked for are dropped before they
    # are downloaded rather than after.
    time_start = astropy.time.Time(time_start)

    # Without `limit`, which has nothing to say in a branch which returns one
    # image however it is set, and which would take away the very record this
    # is here to choose between: it narrows the range to make room for its
    # sampling period, and the candidate on the near side of `time_start`
    # falls outside what is left.
    found = sdo.hmi.search(
        time_start=time_start,
        time_stop=time_start + _cadence(series),
        series=series,
        segment=segment,
        axis_time=axis_time,
        cache=cache,
    )

    index = _index_nearest(
        time=found["DATE-OBS"],
        time_reference=time_start,
        axis_time=axis_time,
    )
    found = {k: found[k][index] for k in found}

    files = sdo.download(
        urls=found["url"],
        cache=cache,
    )

    return Magnetogram.from_fits(
        path=files,
        keywords=found,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )


def _index_nearest(
    time: na.AbstractScalarArray,
    time_reference: astropy.time.Time,
    axis_time: str,
) -> dict[str, slice]:
    """
    The one image closest in time to the one asked for.

    Expressed as a slice rather than an integer, so that taking it leaves the
    time axis in place with one element on it instead of removing it: an
    image is a sequence of one, not a thing of a different shape.

    Parameters
    ----------
    time
        The time each image was taken.
    time_reference
        The time asked for.
    axis_time
        The logical axis corresponding to changes in time.
    """
    t = astropy.time.Time(np.ravel(time.ndarray))
    index = int(np.argmin(np.abs((t - time_reference).to_value(u.s))))
    return {axis_time: slice(index, index + 1)}
