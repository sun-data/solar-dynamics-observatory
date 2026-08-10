import joblib
from typing import Self, Literal
import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import sdo

__all__ = [
    "Magnetogram",
]


@dataclasses.dataclass(eq=False, repr=False)
class Magnetogram(
    na.FunctionArray[
        na.ExplicitTemporalWcsPositionalVectorArray,
        na.ScalarArray,
    ],
):
    """
    A sequence of HMI images of one quantity.

    HMI observes a single spectral line and reports quantities derived from
    it, so unlike :class:`sdo.aia.Filtergram` there is no wavelength axis:
    an image is a place and a time, and which quantity it is was decided
    when it was asked for.
    """

    axis_time: str = "time"
    """The logical axis corresponding to changes in time."""

    axis_detector_x: str = "detector_x"
    """The logical axis corresponding to changes in detector :math:`x`-coordinate."""

    axis_detector_y: str = "detector_y"
    """The logical axis corresponding to changes in detector :math:`y`-coordinate."""

    @classmethod
    def from_time_range(
        cls,
        time_start: str | astropy.time.Time,
        time_stop: str | astropy.time.Time,
        series: Literal["hmi.M_45s", "hmi.M_720s"] = "hmi.M_45s",
        segment: str = "magnetogram",
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        limit: None | int = None,
        cache: None | str | joblib.Memory = sdo.memory,
    ) -> Self:
        """
        Given a time range, download the corresponding HMI images.

        Parameters
        ----------
        time_start
            The start time of the search period.
        time_stop
            The end time of the search period.
        series
            The data series to download.
        segment
            The segment of the series to download, which is the quantity
            being asked for.
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
        cache
            The location to cache the results of this method.
            If not provided, the default cache location, :attr:`sdo.memory`
            is used. If :obj:`None`, no caching is performed, and if `cache`
            is a pathlike, a new cache is created at that location.
        """

        found = sdo.hmi.search(
            time_start=time_start,
            time_stop=time_stop,
            series=series,
            segment=segment,
            axis_time=axis_time,
            limit=limit,
            cache=cache,
        )

        files = sdo.download(
            urls=found["url"],
            cache=cache,
        )

        return cls.from_fits(
            path=files,
            keywords=found,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    @classmethod
    def from_fits(
        cls,
        path: pathlib.Path | na.ScalarArray[pathlib.Path],
        keywords: dict[str, na.ScalarArray],
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> Self:
        """
        Given a single FITS file or an array of FITS files, construct a
        :class:`Magnetogram`.

        Parameters
        ----------
        path
            A single FITS file or an array of FITS files to load.
        keywords
            The keywords describing the files, from :func:`sdo.hmi.search`.
            The files themselves carry none, see that function.
        axis_time
            The logical axis corresponding to changes in time.
        axis_detector_x
            The logical axis corresponding to changes in detector
            :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector
            :math:`y`-coordinate.
        """

        path = na.asarray(path)
        shape_base = path.shape

        # The image is in the first extension, since JSOC serves it
        # compressed, and its shape is all the file has to say.
        with astropy.io.fits.open(path.ndarray.item(0)) as hdul_prototype:
            hdu_prototype = hdul_prototype[_index_data(hdul_prototype)]
            shape_data = hdu_prototype.data.shape

        shape_wcs = {
            axis_detector_y: shape_data[0],
            axis_detector_x: shape_data[1],
        }

        unit = _unit(keywords)
        unit_position = _unit_position(keywords)

        self = cls.empty(
            shape_base=shape_base,
            shape_wcs=shape_wcs,
            unit=unit,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

        for index in path.ndindex():
            file = path[index].ndarray

            with astropy.io.fits.open(
                name=file,
                output_verify="silentfix",
            ) as hdul:
                hdu = hdul[_index_data(hdul)]

                self.outputs[index] = na.ScalarArray(
                    ndarray=hdu.data << unit,
                    axes=tuple(shape_wcs),
                )

            time = astropy.time.Time(keywords["DATE-OBS"][index].ndarray).jd
            self.inputs.time[index] = time

            crval = self.inputs.crval
            crval.position.x[index] = keywords["CRVAL1"][index].ndarray * unit_position
            crval.position.y[index] = keywords["CRVAL2"][index].ndarray * unit_position

            crpix = self.inputs.crpix
            crpix.components[axis_detector_x][index] = keywords["CRPIX1"][index].ndarray
            crpix.components[axis_detector_y][index] = keywords["CRPIX2"][index].ndarray

            cdelt = self.inputs.cdelt
            cdelt.position.x[index] = keywords["CDELT1"][index].ndarray * unit_position
            cdelt.position.y[index] = keywords["CDELT2"][index].ndarray * unit_position

            # HMI is mounted upside down, so `CROTA2` is near 180 degrees
            # rather than near zero. The rotation is given as an angle where
            # AIA gives a matrix, so the matrix is built from it.
            angle = keywords["CROTA2"][index].ndarray * u.deg
            cos = np.cos(angle).value
            sin = np.sin(angle).value

            pc = self.inputs.pc
            pc.position.x.components[axis_detector_x][index] = cos
            pc.position.x.components[axis_detector_y][index] = -sin
            pc.position.y.components[axis_detector_x][index] = sin
            pc.position.y.components[axis_detector_y][index] = cos

        t = astropy.time.Time(
            val=self.inputs.time.ndarray,
            format="jd",
        )
        t.format = "isot"
        self.inputs.time.ndarray = t

        return self

    @classmethod
    def empty(
        cls,
        shape_base: dict[str, int],
        shape_wcs: dict[str, int],
        unit: u.UnitBase = u.G,
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> Self:
        """
        Create an empty :class:`Magnetogram`.

        Parameters
        ----------
        shape_base
            The shape of the result excluding the axes handled by WCS.
        shape_wcs
            The shape of the axes handled by WCS.
        unit
            The unit of the quantity being represented.
        axis_time
            The logical axis corresponding to changes in time.
        axis_detector_x
            The logical axis corresponding to changes in detector
            :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector
            :math:`y`-coordinate.
        """

        inputs = na.ExplicitTemporalWcsPositionalVectorArray(
            time=na.ScalarArray.zeros(shape_base),
            crval=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.empty(shape_base) << u.arcsec,
                    y=na.ScalarArray.empty(shape_base) << u.arcsec,
                ),
            ),
            crpix=na.CartesianNdVectorArray(
                components={
                    axis_detector_x: na.ScalarArray.empty(shape_base),
                    axis_detector_y: na.ScalarArray.empty(shape_base),
                }
            ),
            cdelt=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.empty(shape_base) << u.arcsec,
                    y=na.ScalarArray.empty(shape_base) << u.arcsec,
                ),
            ),
            pc=na.PositionalMatrixArray(
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components={
                            axis_detector_x: na.ScalarArray.empty(shape_base),
                            axis_detector_y: na.ScalarArray.empty(shape_base),
                        },
                    ),
                    y=na.CartesianNdVectorArray(
                        components={
                            axis_detector_x: na.ScalarArray.empty(shape_base),
                            axis_detector_y: na.ScalarArray.empty(shape_base),
                        },
                    ),
                ),
            ),
            shape_wcs={a: shape_wcs[a] + 1 for a in shape_wcs},
        )

        shape = na.broadcast_shapes(shape_base, shape_wcs)
        outputs = na.ScalarArray.empty(shape) << unit

        return cls(
            inputs=inputs,
            outputs=outputs,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )


def _index_data(hdul: astropy.io.fits.HDUList) -> int:
    """The extension holding the image, which is not always the primary one."""
    for i, hdu in enumerate(hdul):
        if hdu.data is not None:
            return i
    raise ValueError("no extension of this file holds an image")  # pragma: nocover


def _unit(keywords: dict[str, na.ScalarArray]) -> u.UnitBase:
    """
    The unit of the image, taken from the keywords rather than assumed.

    The quantities HMI reports are in different units from one another, where
    every AIA filtergram is in counts, so this cannot be a constant.
    """
    if "BUNIT" not in keywords:  # pragma: nocover
        return u.dimensionless_unscaled
    bunit = str(keywords["BUNIT"].ndarray.reshape(-1)[0])
    # HMI spells the gauss `Gauss`, which astropy does not know by that name.
    return u.Unit(bunit.replace("Gauss", "G"), parse_strict="silent")


def _unit_position(keywords: dict[str, na.ScalarArray]) -> u.UnitBase:
    """The unit of the plate scale and the tangent point."""
    if "CUNIT1" not in keywords:  # pragma: nocover
        return u.arcsec
    return u.Unit(str(keywords["CUNIT1"].ndarray.reshape(-1)[0]), parse_strict="silent")
