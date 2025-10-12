# WaddingtonPathIntegralDynamics

Waddington Path Integral Dynamics (WPID) is a MATLAB package for fitting dynamical
landscape models directly to high-dimensional single-cell data. This implementation accompanies
"Reconstructing Waddington's Landscape from Data" by Cislo, Delás, Briscoe, and Siggia (2025).

The package provides computational geometry methods for modeling cell-fate dynamics as probability
flows in gene expression space. Core features include:

- **Landscape reconstruction**: Algorithms to fit potential landscapes with minimal free parameters
  from single-cell measurements (flow cytometry, RNA-seq)
- **Dynamical analysis**: Tools for identifying fixed points, unstable manifolds, and basins of
  attraction in high-dimensional data
- **Transition dynamics**: Methods for computing most probable paths and transition matrices
  between cell states
- **Temporal modeling**: Probability distribution evolution under different signaling conditions,
  including landscape interpolation for capturing signal-dependent dynamics

## Installation prerequisites

- MATLAB R2021a or newer. Earlier releases may work, but have not been validated recently.
- Optional: Statistics and Machine Learning Toolbox for distance computations used in density
  estimation utilities.
- Optional: Parallel Computing Toolbox to accelerate larger diffusion map or Monte Carlo workflows.
- Optional: A supported C/C++ compiler if you intend to build the bundled MEX utilities via
  `compile_mex.m`. Building the `External/ngl-beta` bindings enables
  `Utility/PointCloudDensityEstimation/pointCloudDensityEstimation` and
  `Utility/ProximityGraphs/NGL/proximityGraphsNGL`, which the synthetic tutorial calls. Without
  those MEX targets, these features are unavailable. `compile_mex.m` is currently only
  compatible with Linux and MacOS.

## Getting started

Run the annotated workflow in `SyntheticDataExample/Synthetic_Data_Analysis_Script.m` for a guided
walk-through of the pipeline that recreates the heteroclinic flip example in the paper (see the
section **Algorithm Applied to Simulated Data**). The script demonstrates potential landscape
fitting, transition matrix estimation, and probability-density evolution using the core functions in
`Source/`.

## Repository tour

- `Source/`: Core routines for estimating potentials, fitting dynamical landscapes, and evaluating
  transition dynamics.
- `Utility/`: Shared numerical helpers (diffusion maps, density estimation, proximity graphs,
  mesh/point-cloud utilities) leveraged by the main algorithms.
- `SyntheticDataExample/`: End-to-end tutorial replicating the heteroclinic flip example and
  showcasing recommended workflows.
- `Tests/`: MATLAB validation scripts exercising fitting, transition-matrix construction, and time
  scale estimation utilities.
- `External/`: Third-party dependencies such as `ngl-beta`, `gptoolbox`, and colormap utilities
  required for selected features and visualizations.
- `compile_mex.m` and friends: Convenience scripts for building the bundled MEX extensions,
  including the `Utility/ProximityGraphs/NGL` bindings backed by `External/ngl-beta`.

## License

This project is distributed under the terms of the MIT License. See `LICENSE` for the full text.
