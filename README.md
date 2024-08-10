# solar-dynamics-observatory

[![tests](https://github.com/sun-data/solar-dynamics-observatory/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/solar-dynamics-observatory/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/solar-dynamics-observatory/graph/badge.svg?token=9VdGTSq2hT)](https://codecov.io/gh/sun-data/solar-dynamics-observatory)
[![Black](https://github.com/sun-data/solar-dynamics-observatory/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/solar-dynamics-observatory/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/solar-dynamics-observatory/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/solar-dynamics-observatory/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/sdo/badge/?version=latest)](https://sdo.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/solar-dynamics-observatory.svg)](https://badge.fury.io/py/solar-dynamics-observatory)

A Python library to download and analyze data from the NASA Solar Dynamics Observatory (SDO).

This library uses the [Sunpy](https://sunpy.org/) package to download the images
and the [aiapy](https://aiapy.readthedocs.io) package to align and prepare the 
images to be ready for scientific analysis.

While this package uses Sunpy to download the data, it does not represent SDO
images as instances of
[`sunpy.map.Map`](https://docs.sunpy.org/en/stable/generated/api/sunpy.map.Map.html),
it represents images using [`named_arrays.FunctionArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.FunctionArray.html).

## Installation

This package is published on PyPI and can be installed using pip
```bash
pip install solar-dynamics-observatory
```
