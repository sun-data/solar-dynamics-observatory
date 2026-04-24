from typing import Self, Literal
import os
import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
import sunpy.net.attrs
import aiapy.calibrate.utils
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

    @classmethod
    def from_time_range(
        cls,
        time_start: str | astropy.time.Time,
        time_stop: str | astropy.time.Time,
        wavelength: u.Quantity | na.ScalarArray,
        user_email: None | str = None,
        series: Literal["aia.lev1_euv_12s", "aia.lev1_uv_24s"] = "aia.lev1_euv_12s",
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ):
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
        """

        time_start = astropy.time.Time(time_start)
        time_stop = astropy.time.Time(time_stop)

        wavelength = na.as_named_array(wavelength)
        if wavelength.ndim == 0:
            axis_wavelength = "wavelength"
            wavelength = wavelength.add_axes(axis_wavelength)
        elif wavelength.ndim == 1:
            axis_wavelength = wavelength.axes[0]
        else:  # pragma: nocover
            raise ValueError(f"`wavelength` must be 0D or 1D, got {wavelength.shape=}")

        directory = sdo.directory_default
        directory.mkdir(parents=True, exist_ok=True)

        directory_level_1 = directory / "level_1"
        directory_level_15 = directory / "level_15"

        directory_level_15.mkdir(parents=True, exist_ok=True)

        pointing_table = aiapy.calibrate.utils.get_pointing_table(
            source="JSOC",
            time_range=(
                time_start - 12 * u.h,
                time_stop + 12 * u.h,
            ),
        )

        if user_email is None:
            user_email = os.environ["JSOC_EMAIL"]

        time = sunpy.net.attrs.Time(time_start, time_stop)
        notify = sunpy.net.attrs.jsoc.Notify(user_email)
        segment = sunpy.net.attrs.jsoc.Segment("image")

        series = sunpy.net.attrs.jsoc.Series(series)

        files = []

        for w in wavelength.ndindex():

            channel = wavelength[w].ndarray

            search = sunpy.net.Fido.search(
                time,
                series,
                notify,
                sunpy.net.attrs.jsoc.Wavelength(channel),
                segment,
            )

            files_wavelength = sunpy.net.Fido.fetch(
                search,
                path=directory_level_1,
                progress=False,
            )

            files_wavelength = sorted(files_wavelength)

            files_15 = []
            for file in files_wavelength:

                file_15 = directory_level_15 / pathlib.Path(file).name
                files_15.append(file_15)

                if not file_15.is_file():

                    aia_map = sunpy.map.Map(file)
                    aia_map = aiapy.calibrate.update_pointing(
                        smap=aia_map,
                        pointing_table=pointing_table,
                    )
                    # aia_map = aiapy.calibrate.register(aia_map)
                    aia_map.save(file_15)

            files_wavelength = files_15

            files_wavelength = np.array(files_wavelength)

            files_wavelength = na.ScalarArray(files_wavelength, axes=axis_time)

            files.append(files_wavelength)

        files = na.stack(files, axis=axis_wavelength)

        files = files.transpose((axis_time, axis_wavelength))

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

        index_max = {
            axis_detector_x: slice(None, shape_wcs[axis_detector_x]),
            axis_detector_y: slice(None, shape_wcs[axis_detector_y]),
        }

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

            self.outputs[index] = na.ScalarArray(
                ndarray=hdu.data << u.DN,
                axes=tuple(shape_wcs),
            )[index_max]

            time = astropy.time.Time(hdu.header["DATE-OBS"]).jd
            self.inputs.time[index] = time

            wcs = astropy.wcs.WCS(hdu).wcs

            crval = self.inputs.crval
            crval.position.x[index] = wcs.crval[~ix] << u.deg
            crval.position.y[index] = wcs.crval[~iy] << u.deg

            crpix = self.inputs.crpix
            crpix.components[axis_detector_x][index] = wcs.crpix[~ix]
            crpix.components[axis_detector_y][index] = wcs.crpix[~iy]

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
            # timedelta=timedelta,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )
