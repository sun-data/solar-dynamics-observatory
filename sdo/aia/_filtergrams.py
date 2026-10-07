import joblib
from typing import Self, Literal
import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
import named_arrays as na
import sdo

__all__ = [
    "Filtergram",
]


@dataclasses.dataclass(eq=False, repr=False)
class Filtergram(
    na.FunctionArray[
        na.ExplicitTemporalSpectralWcsPositionalVectorArray,
        na.ScalarArray,
    ],
):
    """
    A representation of an AIA image sequence using any number of filters.
    """

    timedelta: u.Quantity | na.AbstractScalar = 0 * u.s
    """The exposure time of each image."""

    axis_time: str = "time"
    """The logical axis corresponding to changes in time."""

    axis_wavelength: str = "wavelength"
    """The logical axis corresponding to changes in wavelength."""

    axis_detector_x: str = "detector_x"
    """The logical axis corresponding to changes in detector :math:`x`-coordinate."""

    axis_detector_y: str = "detector_y"
    """The logical axis corresponding to changes in detector :math:`y`-coordinate."""

    @classmethod
    def from_time_range(
        cls,
        time_start: str | astropy.time.Time,
        time_stop: str | astropy.time.Time,
        wavelength: u.Quantity | na.ScalarArray,
        series: Literal["aia.lev1_euv_12s", "aia.lev1_uv_24s"] = "aia.lev1_euv_12s",
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        limit: None | int = None,
        register: bool = False,
        cache: None | str | joblib.Memory = sdo.memory,
    ):
        """
        Given a time range and a wavelength, download the corresponding
        AIA filtergram.

        Parameters
        ----------
        time_start
            The start time of the search period
        time_stop
            The end time of the search period.
        wavelength
            The wavelengths to download.
            Must be a valid AIA wavelength.
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
        register
            Boolean flag controlling whether the images are registered using
            :func:`aiapy.calibrate.register`, which rotates each image to solar
            north up and scales it to a common plate scale.
        cache
            The location to cache the results of this method.
            If not provided, the default cache location, :attr:`sdo.memory` is used.
            If :obj:`None`, no caching is performed, and if `cache` is a pathlike,
            a new cache is created at that location.
        """

        urls = sdo.aia.urls(
            time_start=time_start,
            time_stop=time_stop,
            wavelength=wavelength,
            series=series,
            axis_time=axis_time,
            limit=limit,
            cache=cache,
        )

        files = sdo.aia.download(
            urls=urls,
            cache=cache,
        )

        files = sdo.aia.prep(
            files=files,
            register=register,
            cache=cache,
        )

        (axis_wavelength,) = set(files.shape) - {axis_time}

        return cls.from_fits(
            path=files,
            wavelength=wavelength,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    @classmethod
    def from_fits(
        cls,
        path: pathlib.Path | na.ScalarArray[pathlib.Path],
        wavelength: u.Quantity | na.ScalarArray,
        axis_time: str = "time",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> Self:
        """
        Given a single FITS file or an array of FITS files with the same OBSID,
        construct a SpectrographObservation object.

        Parameters
        ----------
        path
            A single FITS file or an array of FITS files to load.
        window
            The spectral window to load.
        axis_time
            The logical axis corresponding to changes in time.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        """

        path = na.asarray(path)
        shape_base = path.shape

        hdul_prototype = astropy.io.fits.open(path.ndarray.item(0))

        index_window = 0

        hdu_prototype = hdul_prototype[index_window]
        wcs_prototype = astropy.wcs.WCS(hdu_prototype)

        axes_wcs = list(reversed(wcs_prototype.axis_type_names))

        ix = axes_wcs.index("HPLN")
        iy = axes_wcs.index("HPLT")

        axes_wcs[ix] = axis_detector_x
        axes_wcs[iy] = axis_detector_y

        shape_wcs = wcs_prototype.array_shape
        shape_wcs = {ax: sz for ax, sz in zip(axes_wcs, shape_wcs)}

        self = cls.empty(
            shape_base=shape_base,
            shape_wcs=shape_wcs,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

        for index in path.ndindex():
            file = path[index].ndarray

            hdul = astropy.io.fits.open(
                name=file,
                output_verify="silentfix",
            )
            hdu = hdul[index_window]

            data = na.ScalarArray(
                ndarray=hdu.data << u.DN,
                axes=tuple(shape_wcs),
            )

            # Registration with aiapy leaves some channels a pixel or two
            # smaller than others, with the center of the Sun at the center of
            # each, so each image is centered in the array of the first, and
            # its reference pixel is moved with it. That keeps its coordinates
            # and puts every registered channel on the same grid.
            offset = {a: (shape_wcs[a] - data.shape[a]) // 2 for a in shape_wcs}
            index_out: dict[str, int | slice] = dict(index)
            index_in: dict[str, slice] = dict()
            for a in shape_wcs:
                num = min(shape_wcs[a], data.shape[a])
                index_out[a] = slice(max(offset[a], 0), max(offset[a], 0) + num)
                index_in[a] = slice(max(-offset[a], 0), max(-offset[a], 0) + num)
            if any(offset.values()):
                self.outputs[index] = np.nan * u.DN
            self.outputs[index_out] = data[index_in]

            time = astropy.time.Time(hdu.header["DATE-OBS"]).jd
            self.inputs.time[index] = time

            self.timedelta[index] = hdu.header["EXPTIME"] * u.s

            wcs = astropy.wcs.WCS(hdu).wcs

            crval = self.inputs.crval
            crval.position.x[index] = wcs.crval[~ix] << u.deg
            crval.position.y[index] = wcs.crval[~iy] << u.deg

            # One less than the FITS keyword, which counts pixels from one
            # where :class:`named_arrays.AbstractWcsVector` counts them from
            # zero, as in :mod:`sdo.hmi`. Without this every image sits one
            # pixel, 0.6 arcseconds, from where it belongs.
            crpix = self.inputs.crpix
            crpix.components[axis_detector_x][index] = (
                wcs.crpix[~ix] - 1 + offset[axis_detector_x]
            )
            crpix.components[axis_detector_y][index] = (
                wcs.crpix[~iy] - 1 + offset[axis_detector_y]
            )

            cdelt = self.inputs.cdelt
            cdelt.position.x[index] = wcs.cdelt[~ix] << u.deg
            cdelt.position.y[index] = wcs.cdelt[~iy] << u.deg

            pc_wcs = wcs.get_pc()

            pc = self.inputs.pc
            pc.position.x.components[axis_detector_x][index] = pc_wcs[~ix, ~ix]
            pc.position.x.components[axis_detector_y][index] = pc_wcs[~ix, ~iy]
            pc.position.y.components[axis_detector_x][index] = pc_wcs[~iy, ~ix]
            pc.position.y.components[axis_detector_y][index] = pc_wcs[~iy, ~iy]

        t = astropy.time.Time(
            val=self.inputs.time.ndarray,
            format="jd",
        )
        t.format = "isot"
        self.inputs.time.ndarray = t

        self.inputs.wavelength = wavelength

        return self

    @classmethod
    def empty(
        cls,
        shape_base: dict[str, int],
        shape_wcs: dict[str, int],
        axis_time: str = "time",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> Self:
        """
        Create an empty :class:`Filtergrams` object.

        Parameters
        ----------
        shape_base
            The shape of the result excluding the axes handled by WCS.
        shape_wcs
            The shape of the axes handled by WCS.
        axis_time
            The logical axis corresponding to changes in time.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        """

        inputs = na.ExplicitTemporalSpectralWcsPositionalVectorArray(
            time=na.ScalarArray.zeros(shape_base),
            wavelength=na.ScalarArray.zeros(shape_base) << u.AA,
            crval=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.empty(shape_base) << u.arcsec,
                    y=na.ScalarArray.empty(shape_base) << u.arcsec,
                ),
            ),
            crpix=na.CartesianNdVectorArray(
                components=dict(
                    detector_x=na.ScalarArray.empty(shape_base),
                    detector_y=na.ScalarArray.empty(shape_base),
                )
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
                        components=dict(
                            detector_x=na.ScalarArray.empty(shape_base),
                            detector_y=na.ScalarArray.empty(shape_base),
                        ),
                    ),
                    y=na.CartesianNdVectorArray(
                        components=dict(
                            detector_x=na.ScalarArray.empty(shape_base),
                            detector_y=na.ScalarArray.empty(shape_base),
                        ),
                    ),
                ),
            ),
            shape_wcs={a: shape_wcs[a] + 1 for a in shape_wcs},
        )

        shape = na.broadcast_shapes(shape_base, shape_wcs)
        outputs = na.ScalarArray.empty(shape) << u.DN

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=na.ScalarArray.zeros(shape_base) << u.s,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )
