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

    axis_time: str = "time"
    """The logical axis corresponding to changes in time."""

    axis_wavelength: str = "wavelength"
    """The logical axis corresponding to changes in wavelength."""

    axis_detector_x: str = "detector_x"
    """The logical axis corresponding to changes in detector :math:`x`-coordinate."""

    axis_detector_y: str = "detector_y"
    """The logical axis corresponding to changes in detector :math:`y`-coordinate."""

    # Declared after the axes, so that they keep the positions they had
    # before it, and made anew for each instance, so that changing one in
    # place leaves the others alone.
    timedelta: u.Quantity | na.AbstractScalar = dataclasses.field(
        default_factory=lambda: 0 * u.s,
    )
    """The exposure time of each image."""

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
        Given a single FITS file or an array of FITS files,
        construct a :class:`Filtergram` object.

        Parameters
        ----------
        path
            A single FITS file or an array of FITS files to load.
        wavelength
            The channel of each image.
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

        index_window = 0

        with astropy.io.fits.open(path.ndarray.item(0)) as hdul_prototype:
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

            # Read whole rather than mapped into memory, since a mapped file
            # stays open as long as its data does.
            with astropy.io.fits.open(
                name=file,
                output_verify="silentfix",
                memmap=False,
            ) as hdul:
                hdu = hdul[index_window]
                header = hdu.header
                wcs = astropy.wcs.WCS(hdu).wcs

                data = na.ScalarArray(
                    ndarray=hdu.data << u.DN,
                    axes=tuple(shape_wcs),
                )

            # Registration with aiapy leaves a channel 4094 pixels across
            # instead of 4096 when its plate scale is finer than 0.6 arcseconds
            # (LM-SAL/aiapy#384), with the center of the Sun at the center of
            # the image either way. So each image is centered in the array of
            # the first, and its reference pixel is moved with it. That keeps
            # its coordinates and puts every registered channel on the same
            # grid. A size which differed by an odd number of pixels, which
            # registration does not leave, would put an image half a pixel off
            # that grid, though still at its own coordinates.
            offset = {a: (shape_wcs[a] - data.shape[a]) // 2 for a in shape_wcs}
            index_out: dict[str, int | slice] = dict(index)
            index_in: dict[str, slice] = dict()
            for a in shape_wcs:
                num = min(shape_wcs[a], data.shape[a])
                index_out[a] = slice(max(offset[a], 0), max(offset[a], 0) + num)
                index_in[a] = slice(max(-offset[a], 0), max(-offset[a], 0) + num)
            if data.shape != shape_wcs:
                self.outputs[index] = np.nan * u.DN
            self.outputs[index_out] = data[index_in]

            time = astropy.time.Time(header["DATE-OBS"]).jd
            self.inputs.time[index] = time

            self.timedelta[index] = header["EXPTIME"] * u.s

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

    def resample(
        self,
        position: na.AbstractCartesian2dVectorArray,
        axis: tuple[str, str],
        fwhm: u.Quantity | na.AbstractScalar,
        fwhm_aia: u.Quantity | na.AbstractScalar = 1.3 * u.arcsec,
        uncertainty: None | u.Quantity | na.AbstractScalar = None,
        seed: None | int = 0,
    ) -> tuple[na.AbstractScalar, na.AbstractScalar]:
        """
        Blur these images to a coarser resolution and sum them onto other
        pixels, such as those of another instrument,
        along with their uncertainty.

        The images are blurred by a Gaussian whose width at half its maximum
        is the difference in quadrature between `fwhm` and `fwhm_aia`,
        so that a point source is as wide as the other instrument sees it,
        and then summed onto the new pixels by conservative regridding,
        which keeps the total signal.

        The variance of each pixel is summed onto the new pixels the same
        way, which alone would overestimate the noise,
        since the blur averages neighboring pixels together,
        and the pixels which straddle two new pixels are split between them.
        How much both average the noise down is found by blurring and
        resampling white noise of unit variance,
        and the summed variance of each image is scaled by it.

        Parameters
        ----------
        position
            The helioprojective coordinates of the vertices of the new pixels.
        axis
            The two logical axes of `position` along which the new pixels are
            arranged.
        fwhm
            The full width at half maximum of the point-spread function of the
            new pixels, which must not be smaller than `fwhm_aia`.
        fwhm_aia
            The full width at half maximum of the core of the point-spread
            function of AIA.
        uncertainty
            The one-sigma uncertainty of each pixel of these images.
            If :obj:`None` (the default), :func:`sdo.aia.uncertainty` of the
            images is used.
        seed
            The seed of the white noise which finds how much the noise is
            averaged down.
            If :obj:`None`, the noise and the uncertainty differ a little from
            one call to the next.

        Returns
        -------
        counts
            The signal of each new pixel, in DN.
        uncertainty
            The one-sigma uncertainty of `counts`.

        Examples
        --------
        Sum a registered image of AIA 171 Å onto pixels of 1 arcsecond, as if
        it were seen at a resolution of 2 arcseconds.

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            import astropy.units as u
            import astropy.visualization
            import named_arrays as na
            import sdo

            aia = sdo.aia.open(
                time_start="2021-09-23T06:00",
                wavelength=171 * u.AA,
                register=True,
            )
            aia = aia[dict(
                time=0,
                wavelength=0,
                detector_x=slice(1900, 2200),
                detector_y=slice(1900, 2200),
            )]

            position = na.Cartesian2dVectorArray(
                x=na.linspace(-60, 60, axis="x", num=121) * u.arcsec,
                y=na.linspace(-60, 60, axis="y", num=121) * u.arcsec,
            )

            counts, uncertainty = aia.resample(
                position=position,
                axis=("x", "y"),
                fwhm=2 * u.arcsec,
            )

            with astropy.visualization.quantity_support():
                fig, ax = plt.subplots(constrained_layout=True)
                na.plt.pcolormesh(position, C=counts, ax=ax)
                ax.set_aspect("equal")
        """
        axis_detector = (self.axis_detector_x, self.axis_detector_y)

        if uncertainty is None:
            uncertainty = sdo.aia.uncertainty(
                counts=self.outputs,
                wavelength=self.inputs.wavelength,
            )

        difference = np.square(fwhm) - np.square(fwhm_aia)
        if np.any(difference < 0):
            raise ValueError(
                f"`fwhm`, {fwhm}, must not be smaller than `fwhm_aia`, {fwhm_aia}."
            )
        sigma = np.sqrt(difference) / (2 * np.sqrt(2 * np.log(2)))

        def blur(a: na.AbstractScalar) -> na.AbstractScalar:
            if np.all(sigma == 0):
                return a
            cdelt = self.inputs.cdelt.position
            scale = (np.abs(cdelt.x).mean() + np.abs(cdelt.y).mean()) / 2
            scale = scale.to(u.arcsec)
            # out to four times the width of the Gaussian, beyond which it is
            # less than a ten-thousandth of its peak
            ratio = na.as_named_array(4 * sigma / scale).to(u.dimensionless_unscaled)
            num = int(np.ceil(ratio.max().ndarray.value))
            x = na.arange(-num, num + 1, axis=self.axis_detector_x) * scale
            y = na.arange(-num, num + 1, axis=self.axis_detector_y) * scale
            kernel = np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            kernel = kernel / kernel.sum(axis_detector)
            return na.convolve(a, kernel, axis=axis_detector)

        weights = na.regridding.weights(
            coordinates_input=self.inputs.position,
            coordinates_output=position,
            axis_input=axis_detector,
            axis_output=axis,
            method="conservative",
        )

        def regrid(a: na.AbstractScalar) -> na.AbstractScalar:
            return na.regridding.regrid_from_weights(*weights, values_input=a)

        counts = regrid(blur(self.outputs))
        variance = regrid(np.square(uncertainty))

        shape = self.outputs.shape
        noise = na.random.normal(0, 1, shape_random=shape, seed=seed)
        noise = regrid(blur(noise))
        ones = regrid(na.ScalarArray.ones(shape))
        reduction = np.square(noise).mean(axis) / ones.mean(axis)

        return counts, np.sqrt(reduction * variance)

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
