Halo Histories vs. Galaxy Properties at z=0 IV:
===============================================
The Properties of Quenched Central Galaxies
===========================================

This repository contains the code for the analysis in the paper
[arxiv link](arxiv.org)

The code is primarily written in Julia, with some external python libraries
(e.g. [`halotools`](https://github.com/astropy/halotools)) and can be found in the `src` directory.

The workflow of the analysis from the paper can be followed in the IJulia
notebooks which are in the `notebooks` directory. The order of the notebooks
goes as:

- `observations.ipynb` shows a run through of the data reduction and plotting
- `galaxy_mass_model.ipynb` finds the best fit parameter values of the
stellar mass-halo mass relation for the galaxy samples using the EMERGE
model from Moster et al. 2017
- `mass_delta_mocks.ipynb` shows code for creating mock galaxy samples from
a halo catalogue with stellar mass and environment density distributions that
derive from fits of stellar mass and halo occupation models which were fitted
using the data samples.
