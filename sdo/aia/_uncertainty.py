from typing import cast
import numpy as np
import astropy.units as u
import astropy.table
import aiapy.calibrate
import named_arrays as na

__all__ = [
    "uncertainty",
]


def uncertainty(
    counts: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.AbstractScalar,
    include_chianti: bool = False,
    include_eve: bool = False,
    include_preflight: bool = False,
    error_table: None | astropy.table.QTable = None,
) -> na.ScalarArray:
    """
    The one-sigma uncertainty of AIA images, as estimated by
    :func:`aiapy.calibrate.estimate_error`.

    That is the uncertainty SolarSoft estimates with ``aia_bp_estimate_error``:
    the shot noise of the photons, the read noise, the uncertainty of the dark
    subtraction, and the error of quantization and of compression, with the
    optional systematic uncertainties of the calibration on top.

    Negative counts, which the read noise leaves in faint pixels, are taken as
    zero, which leaves only the noise which does not depend on the signal.
    :func:`aiapy.calibrate.estimate_error` would return NaN for them.

    Parameters
    ----------
    counts
        The signal in each pixel, in DN, as in :attr:`Filtergram.outputs`.
        Not divided by the exposure time, since the shot noise depends on how
        many photons were counted.
    wavelength
        The channel of each image in `counts`, which may vary along any of
        its axes, as in :attr:`Filtergram.inputs.wavelength`.
    include_chianti
        Whether to add the uncertainty of the atomic data in the temperature
        response, as is often done for DEMs.
    include_eve
        Whether to add the uncertainty of the normalization to SDO/EVE.
    include_preflight
        Whether to add the uncertainty of the preflight calibration.
    error_table
        The table of the parameters of the noise of each channel. If
        :obj:`None` (the default), it is fetched with
        :func:`aiapy.calibrate.utils.get_error_table`.

    Examples
    --------
    The relative uncertainty of an image in each channel.

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na
        import sdo

        images = sdo.aia.open(
            "2024-05-10T17:59:45",
            wavelength=na.ScalarArray([171, 94] * u.AA, axes="wavelength"),
        )

        error = sdo.aia.uncertainty(images.outputs, images.inputs.wavelength)

        (error / images.outputs).median(("detector_x", "detector_y"))
    """
    counts = na.as_named_array(counts)
    wavelength = na.as_named_array(wavelength)

    shape = na.shape_broadcasted(counts, wavelength)
    shape_wavelength = na.shape(wavelength)

    result = na.ScalarArray.zeros(shape) << u.DN

    for index in na.ndindex(shape_wavelength):
        channel = cast(na.ScalarArray, wavelength[index]).ndarray
        c = cast(na.ScalarArray, na.broadcast_to(counts, shape)[index])
        value = u.Quantity(c.ndarray).to(u.DN)
        error = aiapy.calibrate.estimate_error(
            np.maximum(value, 0 * u.DN) / u.pix,
            channel,
            include_preflight=include_preflight,
            include_eve=include_eve,
            include_chianti=include_chianti,
            error_table=error_table,
        )
        result[index] = na.ScalarArray(error * u.pix, axes=c.axes)

    return result
