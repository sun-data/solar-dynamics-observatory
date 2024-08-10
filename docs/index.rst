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


Bibliography
============

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
