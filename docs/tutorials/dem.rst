Invert AIA images for a DEM
===========================

This tutorial finds the differential emission measure (DEM) of an active
region, how much plasma there is at each temperature along the line of sight
through each pixel, from six AIA images :cite:p:`Lemen2012`. It uses the
temperature response of AIA as SolarSoft computes it :cite:p:`Boerner2012`,
from :func:`sdo.aia.temperature_response`, the uncertainty of each pixel from
:func:`sdo.aia.uncertainty`, and the regularized inversion of
:cite:t:`Plowman2020`, :func:`utu.dem.plowman`.

The region is NOAA active region 13664 on 2024 May 10.


Download the images
-------------------

The six channels used for DEMs, registered, so that they share one grid of
0.6 arcsecond pixels with the center of the Sun at its center.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import matplotlib.colors
    import numpy as np
    import astropy.units as u
    import astropy.visualization
    import sunpy.visualization.colormaps
    import named_arrays as na
    import sdo
    import utu

    channels = na.ScalarArray(
        ndarray=[94, 131, 171, 193, 211, 335] * u.AA,
        axes="wavelength",
    )

    images = sdo.aia.open(
        time_start="2024-05-10T17:59:45",
        wavelength=channels,
        register=True,
    )

    # one image in each channel, so the axis of time can go
    images = images[dict(time=0)]

    # the exposure time of each image
    images.timedelta

Select a six-arcminute box around the active region, which is on the western
half of the disk.

.. jupyter-execute::

    region = images[dict(
        detector_x=slice(2840, 3440),
        detector_y=slice(1325, 1925),
    )]

    # the corners of the pixels, which registration makes the same in
    # every channel
    position = region.inputs.position[dict(wavelength=0)]

    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=(9, 6.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for i, ax in enumerate(axs.flat):
            index = dict(wavelength=i)
            channel = int(channels[index].ndarray.value)
            rate = (region.outputs[index] / region.timedelta[index]).value
            na.plt.pcolormesh(
                position,
                C=rate,
                ax=ax,
                cmap=f"sdoaia{channel}",
                norm=matplotlib.colors.PowerNorm(
                    gamma=0.5,
                    vmin=0,
                    vmax=rate.percentile(99.5).ndarray,
                ),
            )
            ax.set_title(f"{channel} Å")
            ax.set_aspect("equal")
        fig.supxlabel("helioprojective $x$ (arcsec)")
        fig.supylabel("helioprojective $y$ (arcsec)")


Intensities and uncertainties
-----------------------------

The temperature response gives the signal per pixel per second, so the
images are divided by their exposure times and are per pixel as well.

The uncertainty is that of :func:`aiapy.calibrate.estimate_error`: the shot
noise of the photons and the noise of the camera. It is estimated from the
counts before they are divided by the exposure time, since the shot noise
depends on how many photons were counted.

.. jupyter-execute::

    counts = region.outputs

    intensity = counts / region.timedelta / u.pix

    uncertainty = sdo.aia.uncertainty(
        counts=counts,
        wavelength=region.inputs.wavelength,
    )
    uncertainty = uncertainty / region.timedelta / u.pix

    # the median signal-to-noise ratio of each channel
    (intensity / uncertainty).median(("detector_x", "detector_y"))


Temperature response
--------------------

The response of each channel at the time of the observation, normalized to
SDO/EVE, which is what ``aia_get_response(/temp, /dn, /evenorm,
timedepend_date=...)`` returns in SolarSoft. The DEM is found at the
temperatures of the response, so it is cut to the range the six channels can
constrain, :math:`\log_{10} T` from 5.5 to 7.5.

.. jupyter-execute::

    response = sdo.aia.temperature_response(
        wavelength=channels,
        time=region.inputs.time.ndarray.min(),
        eve=True,
    )

    response = response[dict(temperature=slice(30, 71))]

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            response.inputs,
            response.outputs,
            ax=ax,
            axis="temperature",
            label=na.ScalarArray(
                ndarray=np.array([f"{c:.0f} Å" for c in channels.ndarray.value]),
                axes="wavelength",
            ),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-28, 1e-23)
        ax.set_xlabel(f"temperature ({response.inputs.unit:latex_inline})")
        ax.set_ylabel(f"response ({response.outputs.unit:latex_inline})")
        ax.legend()


Invert
------

Every axis of the intensities other than the channel is a separate
inversion, so this finds the DEM of each of the 360,000 pixels of the
region, on every core.

.. jupyter-execute::

    dem, chi2 = utu.dem.plowman(
        intensity=intensity,
        uncertainty=uncertainty,
        response=response,
        axis_channel="wavelength",
        axis_temperature="temperature",
    )

    # the median reduced chi squared, which the inversion aims at one
    chi2.median()

The emission measure in four ranges of temperature, the integral of the DEM
over each.

.. jupyter-execute::

    logt = np.log10(dem.inputs / u.K)
    step = 0.05

    ranges = [(5.6, 6.0), (6.0, 6.3), (6.3, 6.6), (6.6, 7.0)]

    em = {
        r: (dem.outputs * step).sum("temperature", where=(logt >= r[0]) & (logt < r[1]))
        for r in ranges
    }

    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(8, 8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for ax, r in zip(axs.flat, ranges):
            mesh = na.plt.pcolormesh(
                position,
                C=em[r].value,
                ax=ax,
                cmap="inferno",
                norm=matplotlib.colors.LogNorm(vmin=1e26, vmax=1e29),
            )
            ax.set_title(rf"$\log_{{10}} T$ = {r[0]} to {r[1]}")
            ax.set_aspect("equal")
        fig.supxlabel("helioprojective $x$ (arcsec)")
        fig.supylabel("helioprojective $y$ (arcsec)")
        fig.colorbar(
            mesh.ndarray.item(),
            ax=axs,
            label=f"emission measure ({em[r].unit:latex_inline})",
        )

The whole DEM in one false-color image, made with :mod:`named_arrays.colorsynth`.
Temperature is mapped onto the visible spectrum, from violet for the coolest
plasma to red for the hottest, and each pixel is colored as if the DEM were
the spectrum of the light it emits. Every temperature shares one scale, up
to the 99.5th percentile of the whole DEM, so the brightness of a pixel is how
much plasma it has and the color is at what temperature. Most of the plasma
of the region is at 2 to 3 MK, which comes out green, and the loops at its
core are hotter, yellow to red. Only the temperatures the six channels
constrain are shown, :math:`\log_{10} T` from 5.6 to 7.2.

.. jupyter-execute::

    constrained = dem[dict(temperature=slice(2, 35))]

    # in units of 10^28 cm^-5, so that the colorbar needs no offset
    dem_28 = (constrained.outputs / (1e28 / u.cm**5)).to(u.dimensionless_unscaled)

    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(8, 7),
            gridspec_kw=dict(width_ratios=[0.9, 0.1]),
            constrained_layout=True,
        )
        colorbar = na.plt.rgbmesh(
            np.log10(constrained.inputs / u.K),
            position,
            C=dem_28,
            axis_wavelength="temperature",
            ax=axs[0],
            norm=np.sqrt,
            vmin=0,
            vmax=np.nanpercentile(dem_28, q=99.5),
        )
        na.plt.pcolormesh(
            C=colorbar,
            axis_rgb="temperature",
            ax=axs[1],
        )
        axs[0].set_aspect("equal")
        axs[0].set_xlabel("helioprojective $x$ (arcsec)")
        axs[0].set_ylabel("helioprojective $y$ (arcsec)")
        axs[1].set_xlabel(r"DEM ($10^{28}\,\mathrm{cm^{-5}}$)")
        axs[1].set_ylabel(r"$\log_{10} T$")
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")

The DEM of the pixel with the most plasma above 4 MK, and of the one with the
most below 1 MK.

.. jupyter-execute::

    hot = np.argmax(em[(6.6, 7.0)], axis=("detector_x", "detector_y"))
    cool = np.argmax(em[(5.6, 6.0)], axis=("detector_x", "detector_y"))

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(constrained_layout=True)
        for name, index in [("hot", hot), ("cool", cool)]:
            na.plt.plot(
                dem.inputs,
                dem.outputs[index],
                ax=ax,
                axis="temperature",
                label=f"{name}, $\\chi^2$ = {chi2[index].ndarray:.2f}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"temperature ({dem.inputs.unit:latex_inline})")
        ax.set_ylabel(rf"DEM ({dem.outputs.unit:latex_inline} per unit $\log_{{10}} T$)")
        ax.legend()
