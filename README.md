# RCWA

A Python implementation of **Rigorous Coupled-Wave Analysis (RCWA)** for electromagnetic simulations of periodic structures, with a focus on fast optical force calculations.

## Overview

RCWA is a frequency-domain method for solving Maxwell's equations in layered periodic media. This package provides:

- **Arbitrary periodic geometries** — define structured layers through permittivity and permeability distributions
- **Multi-mode expansion** — control the number of Fourier harmonics for accuracy vs. speed trade-offs
- **Optical force simulation** — efficient computation of radiation pressure and gradient forces on periodic structures
- **Flexible material support** — handles complex, dispersive, and anisotropic media

## Requirements

- `numpy`
- `scipy`

## Installation

```bash
pip install -e .
```

## Quick Start

See [examples/how-to-use.md](examples/how-to-use.md) and [examples/mvtest.py](examples/mvtest.py) for usage examples.

---

# Citation
If you use this software in your research or projects, please cite the following paper:
```bibtex
@article{...,
doi = {10.1088/2040-8986/ad8c58},
url = {https://dx.doi.org/10.1088/2040-8986/ad8c58},
year = {2024},
month = {nov},
publisher = {IOP Publishing},
volume = {26},
number = {12},
pages = {125104},
author = {Bo Gao and Henkjan Gersen and Simon Hanna},
title = {On the suitability of rigorous coupled-wave analysis for fast optical force simulations},
journal = {Journal of Optics},
}
```
We appreciate your support in citing our work!
