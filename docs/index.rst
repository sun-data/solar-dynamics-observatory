Introduction
============

The `Solar Dynamics Observatory <https://sdo.gsfc.nasa.gov>`_ (SDO)
:cite:p:`Pesnell2012` is a NASA satellite which has been continuously
observing the Sun since 2010.

This library uses the :mod:`sunpy` package to download SDO images
and the :mod:`aiapy` package to align and prepare the
images to be ready for scientific analysis.

While this package uses Sunpy to download the data, it does not represent SDO
images as instances of :obj:`sunpy.map.Map`
it represents images using :class:`named_arrays.FunctionArray`.

Installation
============

This package is published to PyPI and can be installed using pip.

.. code-block::

    pip install solar-dynamics-observatory

API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    sdo


Examples
========

Download and display an AIA image

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.visualization
    import named_arrays as na
    import sdo

    images = sdo.aia.Filtergram.from_time_range(
        time_start="2019-09-30T00:00",
        time_stop="2019-09-30T00:00:01",
        wavelength=na.ScalarArray([304] * u.AA, "wavelength"),
        user_email="roytsmart@gmail.com"
    )

    index = {images.axis_time: 0, images.axis_wavelength: 0}

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots()
        na.plt.pcolormesh(
            images.inputs.position.x,
            images.inputs.position.y,
            C=images.outputs.value[index],
            vmin=0,
            vmax=images.outputs.value.percentile(99.9)
        )


Bibliography
============

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
