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


