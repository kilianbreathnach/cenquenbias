Halo Histories vs. Galaxy Properties at z=0 IV:
===============================================
The Properties of Quenched Central Galaxies
===========================================

This repository contains the code for the analysis in the paper
[arxiv link](arxiv.org)

The code is primarily written in Julia, with some external python libraries
(e.g. [`halotools`](https://github.com/astropy/halotools)) and can be found in the `src` directory.
It is compatible with julia v0.7.0 but currently requires the master branches of
Gadfly and Compose.jl packages to make the plots.

The workflow of the analysis from the paper can be followed in the IJulia
notebooks which are in the `notebooks` directory. The order of the notebooks
goes as:

- `observations.ipynb` shows a run through of the data reduction and plotting
- `galaxy_mass.ipynb` finds the best fit parameter values of the
stellar mass-halo mass relation for the galaxy samples using the EMERGE
model from Moster et al. 2017
- `mass_delta_mocks.ipynb` shows code for creating mock galaxy samples from
a halo catalogue with stellar mass and environment density distributions that
derive from fits of stellar mass and halo occupation models which were fitted
using the data samples.
- `optimise_params.ipynb` shows how to run the optimisation on the model to
find fits for the parameters.
- `run_mcmc.jl` in the `src` directory can be used to get samples from the
posteriors.
- The `run_mcmc.py` and `run_pbs_jobs.py` scripts will run the above script
for chosen parameter and observables configurations to generate many samples
at once.
