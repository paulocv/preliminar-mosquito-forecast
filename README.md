# Mosquito relative abundance forecasting – Preliminary version

This repository holds code and a minimally working dataset for the research work: Forecasting the relative abundance of Aedes vector populations to enhance situational awareness for mosquito control operations.

### Preprint (Research Square)
* [Ventura, P. C., Kummer, A. G., Wilke, A. B., Chitturi, J., Hill, M. D., Vasquez, C., ... & Ajelli, M. (2023). Forecasting the relative abundance of Aedes vector populations to enhance situational awareness for mosquito control operations.](https://doi.org/10.21203/rs.3.rs-3464135/v1)

### Source of the Key West relative abundance data:
* Pruszynski, C. A. (2022). Dataset for Aedes aegypti (diptera: Culicidae) and Culex quinquefasciatus (diptera: Culicidae) collections from key West, Florida, USA, 2010–2020. _Data in Brief_, 41, 107907.
  
Licensed under [CC BY 4.0 Deed](https://creativecommons.org/licenses/by/4.0/). The data was aggregated by collection dates, to fit the purposes of this work.

# Code reference

## Dependencies

This project was developed with the following software:
- Python 3.11.5 (with [Conda package manager](https://conda.org/))
  - Details of the conda environment used for the project can be found in [`conda_env_info/`](conda_env_info/).
- Apple clang (C compiler). Version 15.0.0 (clang-1500.0.40.1)
- [Gnu Scientific Library](https://www.gnu.org/software/gsl/) (GSL) – Version 2.7.1

## Historical reproduction number with Monte Carlo Markov Chain (MCMC)

Python script used to preprocess data and run MCMC
- `study_yearly_rt.py`

This script calls the follwing executable: 
- `main_mcmc_rt`

built from the C code in [`rtrend_tools/rt_mcmc/Rt.c`](rtrend_tools/rt_mcmc/Rt.c).

Jupyter notebook used to calculate the average historical R(t) over selected seasons.
- `create_average_rt_ensemble.ipynb`

## Noise patterns – parameter fitting
- `fit_fixed_noise.ipynb`

## Forecast script for one season
- `sweep_forecast_dates.py`

Example:
```sh
python sweep_forecast_dates.py input_params/wis_optimized/keywest-2016.in outputs/wis-calibrated_fas/keywest/latest/optim/2016/
```

# Changelog

## 2024-07-18

Updated the repository after the first review round of the manuscript. 

- Included mosquito collection data divided by trap nights (number of 24h-period observations).
- Made available the aggregated data for the other three study sites: Miami-Dade County, FL, Los Angeles County, CA and Maricopa County, AZ.
- Implemented a naïve model for comparison with our analytical tool. See `baseline_sweep_dates.py`.
- Adjusted methodology to accomodate the mosquito data divided by trap night, which consist of smaller non-integer values compared to the absolute number of specimens.
  - Added upscaling of the data before the MCMC R(t) estimation and the forecast with the renewal equation. Adds also a downscaling back to data per trap night at postprocessing.
  - After upscaling, numbers are clamped to be at least 1, reducing R(t) instabilities.
- The optimization process now uses both WIS and coverage metrics, in a two-step procedure (described in the manuscript).
- Added a revised script for calculating the historic trend of R(t), improving usability and using an optimized [numba](https://numba.pydata.org/) implementation of the MCMC procedure. No changes to the forecast methodology incur from this modification.


## 2023-11-08

Initial version, including minimal code and data for running the analytical tool. Created for the submission of the manuscript.