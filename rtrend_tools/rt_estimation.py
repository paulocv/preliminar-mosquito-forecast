"""
Python interface for the MCMC code in C.
"""
import numpy as np
import os
import pandas as pd
import sys

from rtrend_tools.interpolate import WEEKLEN


# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Custom error class for MCMC in C
class McmcError(Exception):
    pass


class McmcRtEnsemble:
    """
    An ensemble of R(t) time series obtained from MCMC. Carries methods for calculating statistics over it.
    Calculation uses smart getters, avoiding repeated calculations.
    """

    def __init__(self, array: np.ndarray, rolling_width=7, quantile_q=0.025):

        # Constitutive arrays
        self.array = array  # Expected signature: a[exec][i_t]
        self.days = array.shape[1]  # Number of time stamps in the dataset
        self.iters = array.shape[0]  # Nr. of interations from the MCMC process

        self.df = pd.DataFrame(array)  # a[exec][i_t]

        # Calculation parameters
        self.rolling_width = rolling_width
        self.quantile_q = quantile_q

        # TOBE calculated features
        self.avg = None  # Average time series. Np array.  | a[i_t]
        self.rolling_mean = None  # Rolling average ensemble  | a[exec][i_t]
        self.rolling_mean_avg = None  # Mean of the individual rolling averages  | a[i_t]
        self.median: np.ndarray = None  # Median time series  |  a[i_t]
        self.loquant = None  # Lower quantiles  | a[i_t]
        self.hiquant = None  # Upper quantiles  | a[i_t]
        self.roll_loquant = None  # Lower quantile of the rolling average  | a[i_t]
        self.roll_upquant = None
        self.sortd: np.ndarray = None  # Iteration-sorted (np array) R(t) for each t  | a[exec][i_t]

    def get_array(self):
        return self.array

    def get_df(self):
        return self.df

    def get_avg(self, force_calc=False):
        """Return the average time series (average over ensemble of MCMC iterations)."""
        if self.avg is None or force_calc:
            self.avg = np.average(self.array, axis=0)
        return self.avg

    def get_median(self, force_calc=False):
        if self.median is None or force_calc:
            self.median = np.median(self.array, axis=0)
        return self.median

    def get_quantiles(self, force_calc=False):
        """Return the lower and upper quantiles for the raw data (as time series)."""
        if self.loquant is None or self.hiquant is None or force_calc:
            self.loquant = np.quantile(self.array, self.quantile_q, axis=0)
            self.hiquant = np.quantile(self.array, 1.0 - self.quantile_q, axis=0)
        return self.loquant, self.hiquant

    def get_rolling_mean(self, force_calc=False):
        """
        Return the INDIVIDUAL rolling average of each MCMC iteration.
        Signature: a[exec][i_t] (DataFrame)
        """
        if self.rolling_mean is None or force_calc:
            self.rolling_mean: pd.DataFrame = self.df.rolling(self.rolling_width, center=False, axis=1).mean()

        return self.rolling_mean

    def get_rolling_mean_avg(self, force_calc=False):
        """
        Get the AVERAGE time series of the rolling means.
        Signature: a[i_t] (DataFrame)
        """
        if self.rolling_mean_avg is None or force_calc:
            roll = self.get_rolling_mean(force_calc)
            self.rolling_mean_avg = roll.mean(axis=0)

        return self.rolling_mean_avg

    def get_rolling_quantiles(self, force_calc=False):
        """Return the quantiles of the individual rolling averages."""
        if self.roll_loquant is None or self.roll_upquant is None or force_calc:
            roll = self.get_rolling_mean(force_calc)
            self.roll_loquant = roll.quantile(q=self.quantile_q, axis=0)
            self.roll_upquant = roll.quantile(q=1.0 - self.quantile_q, axis=0)

        return self.roll_loquant, self.roll_upquant

    def get_sortd(self):
        """Return the iteration-sorted R(t) ensemble for each time step. This is, for each t, the ensemble of R(t)
        values (iterations) is sorted.
        """
        if self.sortd is None:
            self.sortd = np.sort(self.array, axis=0)

        return self.sortd


def run_mcmc_rt(ct_past, tg_df_roi, nsim=20000, sigma=0.05, seed=0, species="sp",
                in_prefix="mcmc_inputs/species_in/", out_prefix="mcmc_outputs/species_out/",
                log_prefix="mcmc_logs/species_log/", **kwargs):
    """
    Encapsulated interface for running the MCMC R(t) estimator in C and gathering the results.
    """

    # --- Check data shape
    ndays_roi = ct_past.shape[0]
    if tg_df_roi.shape[0] != ndays_roi:
        raise ValueError(f"Hey, ct_past (shape = {ndays_roi}) and tg_df_roi (shape = {tg_df_roi.shape[0]}) must have "
                         f"the same size.")

    # --- Prepare directories and file names
    sn = species.replace(" ", "")  # Eliminates spaces from state names to use in file names.
    for prefix in [in_prefix, out_prefix, log_prefix]:
        os.makedirs(os.path.dirname(prefix), exist_ok=True)  # Make directories if not done yet.

    mcmc_tseries_fname = in_prefix + sn + "_tseries.in"
    mcmc_tgdist_fname = in_prefix + sn + "_tgdist.in"
    mcmc_rtout_fname = out_prefix + sn + "_rt.out"
    mcmc_log_fname = log_prefix + sn + "_err.log"

    # --- Prepare inputs and save to files.
    # Cases (tseries) file
    impcases_array = np.zeros(ndays_roi, dtype=int)
    impcases_array[0] = 1  # Add an imported case
    t_daily = np.arange(ndays_roi)  # Array with sequential numbers. Unused but required in the file.
    np.savetxt(mcmc_tseries_fname, np.array([t_daily, ct_past, impcases_array]).T, fmt="%d")
    # Tg file (tgdist)
    mcmc_tg_array = np.stack([tg_df_roi["Shape"], tg_df_roi["Rate"]], axis=-1)
    np.savetxt(mcmc_tgdist_fname, mcmc_tg_array)

    # --- Call MCMC code
    errcode = os.system(f"./main_mcmc_rt {seed} {nsim} {sigma}"
                        f" {mcmc_tgdist_fname} {mcmc_tseries_fname} > {mcmc_rtout_fname} 2> {mcmc_log_fname}")

    # --- Aftermath
    if errcode:
        msg = f"\n[ERROR {species}] MCMC ERROR CODE = {errcode}. || Please check \"{mcmc_log_fname}\"."
        sys.stderr.write(msg + "\n")
        raise McmcError(msg)

    # --- Load outputs from MCMC (parameters are for stats calculations over the R(t) ensemble)
    return McmcRtEnsemble(np.loadtxt(mcmc_rtout_fname), rolling_width=WEEKLEN, quantile_q=0.025)
