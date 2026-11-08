from typing import Literal
import astropy.units as u
import astropy.table
import named_arrays as na
from ._filtergrams import Filtergram
from ._response import temperature_response
from ._uncertainty import uncertainty as _uncertainty

__all__ = [
    "dem",
]


def dem(
    filtergram: Filtergram,
    uncertainty: None | u.Quantity | na.AbstractScalar = None,
    temperature_min: u.Quantity = 10**5.5 * u.K,
    temperature_max: u.Quantity = 10**7.5 * u.K,
    eve: bool = True,
    chiantifix: bool = False,
    correction_table: None | astropy.table.QTable = None,
    emissivity: Literal["SolarSoft", "CHIANTI 11"] = "SolarSoft",
    smoothness: float = 8,
    chi2_target: float = 1,
    tolerance: float = 0.1,
    steps: tuple[float, float] = (0.1, 0.5),
    iterations_max: int = 100,
    iterations_min: int = 5,
    floor: None | u.Quantity = None,
    axis_temperature: str = "temperature",
) -> tuple[na.FunctionArray[na.AbstractScalar, na.AbstractScalar], na.AbstractScalar]:
    r"""
    Invert AIA images for the differential emission measure (DEM) of each
    pixel, with the method of :cite:t:`Plowman2020`.

    The images are divided by their exposure times, their uncertainty is
    estimated with :func:`uncertainty`, and they are inverted with
    :func:`utu.dem.plowman` against the temperature response of
    :func:`temperature_response` at the time of the earliest image, cut to
    the temperatures from `temperature_min` to `temperature_max`.

    Parameters
    ----------
    filtergram
        The images to invert, one in each channel along
        ``filtergram.axis_wavelength``, such as the six optically thin EUV
        channels, 94, 131, 171, 193, 211, and 335 angstroms. Every other
        axis, such as that of time or of the pixels, is a separate
        inversion.
    uncertainty
        The one-sigma uncertainty of the counts of each image, in DN, as
        :attr:`Filtergram.outputs` is, before they are divided by the
        exposure time. If :obj:`None` (the default), it is that of
        :func:`uncertainty`, the noise of the photons and of the camera.
        An uncertainty given replaces that noise rather than adding to it,
        so a systematic uncertainty is added to :func:`uncertainty` in
        quadrature, and one in DN/s is first multiplied by
        ``filtergram.timedelta``.
    temperature_min
        The lowest temperature of the DEM.
    temperature_max
        The highest temperature of the DEM. The default range,
        :math:`\log_{10} T` from 5.5 to 7.5, is what the six EUV channels
        constrain.
    eve
        Whether to normalize the response to SDO/EVE, which SolarSoft
        recommends together with the correction for the time of the
        observation, see :func:`temperature_response`.
    chiantifix
        Whether to add the empirical correction of the 94 angstrom channel
        for emission missing from CHIANTI, which requires `eve`, see
        :func:`temperature_response`.
    correction_table
        The table of the degradation and EVE normalization of each channel,
        see :func:`temperature_response`. If :obj:`None` (the default), it
        is fetched from SolarSoft.
    emissivity
        The emissivity of the response, see :func:`temperature_response`.
    smoothness
        How many e-foldings per decade of temperature the DEM may change by
        before the regularization resists, see :func:`utu.dem.plowman`.
    chi2_target
        The reduced :math:`\chi^2` the inversion aims for, see
        :func:`utu.dem.plowman`.
    tolerance
        How close to `chi2_target` is close enough, see
        :func:`utu.dem.plowman`.
    steps
        The small and the large fraction of the way to the linearized
        solution to try at each step, see :func:`utu.dem.plowman`.
    iterations_max
        The most steps to take for any one pixel, see
        :func:`utu.dem.plowman`.
    iterations_min
        How long to persist with a pixel whose :math:`\chi^2` has stalled,
        see :func:`utu.dem.plowman`.
    floor
        The least intensity the initial guess assumes in any channel, in
        units convertible to DN/s/pix, see :func:`utu.dem.plowman`.
    axis_temperature
        The name of the axis of temperature of the result.

    Returns
    -------
    dem
        The DEM of each pixel at each temperature of the response, per unit
        :math:`\log_{10} T`, in :math:`\text{cm}^{-5}`, along
        `axis_temperature` and every axis of the images but
        ``filtergram.axis_wavelength``. It is that of :math:`n_e n_H`, the
        convention of the response, and is linear in :math:`\log_{10} T`
        between the temperatures, so the emission measure of each is
        :func:`utu.dem.emission_measure` of it.
    chi2
        The reduced :math:`\chi^2` of each pixel, which is negative where the
        inversion failed, see :func:`utu.dem.plowman`. The pixels to keep are
        those with ``chi2 >= 0``.

    Notes
    -----
    The response is corrected for the degradation of the channels at a
    single time, that of the earliest image, so a series long enough for the
    channels to degrade between its first and last images is better inverted
    a part at a time.

    Examples
    --------
    The DEM of a small region at the center of the disk, from the six
    optically thin EUV channels, averaged over its pixels.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import sdo

        images = sdo.aia.open(
            time_start="2024-05-10T17:59:45",
            wavelength=na.ScalarArray(
                ndarray=[94, 131, 171, 193, 211, 335] * u.AA,
                axes="wavelength",
            ),
            register=True,
        )

        region = images[dict(
            time=0,
            detector_x=slice(1998, 2098),
            detector_y=slice(1998, 2098),
        )]

        dem, chi2 = sdo.aia.dem(region)

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                dem.inputs,
                dem.outputs.mean(("detector_x", "detector_y")),
                ax=ax,
                axis="temperature",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"temperature ({dem.inputs.unit:latex_inline})")
            ax.set_ylabel(f"DEM ({dem.outputs.unit:latex_inline})")
    """
    # utu imports fiasco and matplotlib, which take a second, so it is
    # imported only by those who invert
    import utu

    axis_channel = filtergram.axis_wavelength
    counts = filtergram.outputs
    wavelength = filtergram.inputs.wavelength

    if uncertainty is None:
        uncertainty = _uncertainty(counts=counts, wavelength=wavelength)

    # the response is per pixel per second
    per = filtergram.timedelta * u.pix
    intensity = counts / per
    uncertainty = uncertainty / per

    response = temperature_response(
        wavelength=wavelength,
        time=filtergram.inputs.time.ndarray.min(),
        eve=eve,
        chiantifix=chiantifix,
        axis_wavelength=axis_channel,
        axis_temperature=axis_temperature,
        correction_table=correction_table,
        emissivity=emissivity,
    )
    t = response.inputs
    response = response[(t >= temperature_min) & (t <= temperature_max)]

    return utu.dem.plowman(
        intensity=intensity,
        uncertainty=uncertainty,
        response=response,
        axis_channel=axis_channel,
        axis_temperature=axis_temperature,
        smoothness=smoothness,
        chi2_target=chi2_target,
        tolerance=tolerance,
        steps=steps,
        iterations_max=iterations_max,
        iterations_min=iterations_min,
        floor=floor,
    )
