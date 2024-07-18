"""
Python interface for the MCMC code in C.
"""
import math
import os
import sys

import numba as nb
import numpy as np
import pandas as pd
import scipy

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

    def __init__(
            self, array: np.ndarray, rolling_width=7, quantile_q=0.025,
            date_index=None,
    ):

        # Constitutive arrays
        self.array = array  # Expected signature: a[exec][i_t]
        self.days = array.shape[1]  # Number of time stamps in the dataset
        self.iters = array.shape[0]  # Nr. of interations from the MCMC process

        self.df = pd.DataFrame(array)  # a[exec][i_t]

        # Calculation parameters
        self.rolling_width = rolling_width
        self.quantile_q = quantile_q

        # Includes the date index as columns of the df, if informed
        if date_index is not None:
            # Check integrity
            if date_index.shape[0] != self.days:
                raise ValueError(
                    f"Hey, the shape of `date_index` ({date_index.shape[0]}) "
                    f"must match the "
                    f"number of columns in the array ({self.days}).")

            self.df.columns = date_index

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


def run_mcmc_rt_c(
        ct_past, tg_df_roi, nsim=20000, sigma=0.05, seed=0, species="sp",
        in_prefix="mcmc_inputs/species_in/", out_prefix="mcmc_outputs/species_out/",
        log_prefix="mcmc_logs/species_log/", **kwargs
):
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


# ======================================================================

# NUMBA ENGINE

# ======================================================================


def run_mcmc_rt_numba(
        data_array: np.ndarray,
        tg_shaperate_array: np.ndarray,  # Signat. a[:, 0] = shape, a[:, 1]=rate
        nsim=20000, ntrans=10000, sigma=0.05, seed=0, tg_max=90, i_t0=0,
        tg_is_forward=True,
        **kwargs,
) -> McmcRtEnsemble:
    """
    This wrapper calls the numba MCMC engine to estimate R(t) from past
    incidence data.

    Future Extra features:?
    - Support other Tg distributions, as well as fully specified vectors.
    """
    # p = _nb_gamma_pdf(2.0, tg_shaperate_array[0,0], 1 / tg_shaperate_array[1,0])

    # Checks
    # ------
    # --- Checks data shape
    roi_len = data_array.shape[0]
    if tg_shaperate_array.shape[1] != 2:
        raise ValueError(
            f"Hey, tg_shaperate_array must have 2 columns, but"
            f" {tg_shaperate_array.shape[1]} were found.")
    if tg_shaperate_array.shape[0] != roi_len:
        raise ValueError(
            f"Hey, ct_past (len = {roi_len}) and tg_shaperate_array "
            f"(len = {tg_shaperate_array.shape[0]}) must have the same "
            f"length in the first axis.")

    # --- Checks transient and stationary sizes
    if nsim < ntrans:
        raise ValueError(
            f"nsim = {nsim} must be greater than ntrans = {ntrans}."
        )

    # --- Checks non-negativity of the past data
    neg_mask: np.ndarray = data_array < 0.
    if neg_mask.any():
        num_neg = neg_mask.sum()
        # noinspection PyArgumentList
        min_neg = data_array.min()
        raise ValueError(
            f"Data to MCMC must be non-negative, but {num_neg} negative"
            f" entries were found. Minimum value = {min_neg}."
        )

    # Preprocessing
    # ---------------------

    # --- Pre-Generates the TG distributions for each step
    # -()- Mass-generate, then normalize and transpose
    omega = scipy.stats.gamma.pdf(
        np.arange(tg_max)[:, np.newaxis],
        a=tg_shaperate_array[:, 0],
        scale=1. / tg_shaperate_array[:, 1]
    )

    omega /= omega[1:].sum(axis=0)
    omega = omega.transpose()

    # Engine call
    # -----------
    rt = McmcRtEnsemble(
        _mcmc_numba_engine(
            data_array,
            omega,
            seed, nsim, ntrans, sigma, i_t0,
            tg_is_forward
        )
    )

    return rt


@nb.njit(
    (nb.float64, nb.int32, nb.float64)
)
def _numba_poisson_loglikelihood(r, data_i, past_c):
    """Calculates the natural logarithm of a poisson pmf, with
    parameter r * past_c and evaluated at data_i.

    This is the loglikelihood of explaining the observed incidence
    `data_i` with a reproduction number `r` and past weighted incidence
    `past_c`.
    """
    # Negative reproduction number or data: impossible
    if r < 0. or data_i < 0:
        return -np.inf

    lamb = r * past_c  # Poisson parameter
    return data_i * np.log(lamb) - lamb - math.lgamma(data_i + 1)


@nb.njit(
    (nb.int64[:], nb.float64[:, :], nb.int32, nb.int32, nb.int32,
     nb.float64, nb.int32, nb.boolean)
 )
def _mcmc_numba_engine(data, omega, seed, nsim, ntrans, sigma, i_t0, tg_is_forward):
    """Numba-MCMC to estimate the reproduction number R(t) from
    temporal incidence data.

    This implementation tries to reproduce the algorithm of the C
    version without additional features.

    This engine function is for internal use. You may prefer to call
    `run_mcmc_nb` instead.

    Parameters
    ----------
    data : nb.float64[:]
        The data from which R(t) is calculated.
    omega : nb.float64[:, :]
        The generation time distribution for each time step of the data.
        Signature: a[i_t, i_tg] = p(tg at time t).
        As the generation time cannot be zero, the first entry a[i_t, 0]
        for any time i_t is neglected.
    seed : nb.int32
        Random generator seed.
    nsim : nb.int32
        Total number of MCMC steps to perform (including transient and
        stationary).
    ntrans : nb.int32
        Number of transient MCMC steps to run before storing the
        parameters.
    sigma : nb.float64
        Width of the gaussian step for the R value candidate sampling.
    i_t0 : int
        Estimate R(t) for this time point on.
    tg_is_forward : bool
        Whether the informed generation time (as omega) represents a
        forward distribution (i.e., tg of today is the probability for
        future generated individuals). Converts to backwards if so.

    """
    # Preamble
    # --------
    # * * * Glossary * * *
    # - n_tsteps = number of time points in the input data.
    # - tg_max = maximum generation time to be considered.
    # - i_t0 = first time point in the data to run MCMC for.
    # - n_tsteps_mcmc = effective number of time steps to run MCMC for.

    # Infer number of time steps and max tg from tg distribution
    n_tsteps, tg_max = omega.shape  # Number of time steps in data | maximum gen. time
    n_tsteps_mcmc = n_tsteps - i_t0  # Number of time steps to run MCMC for
    nstat = nsim - ntrans  # Number of MCMC steps in the stationary phase

    # Allocate the resulting array of R values.
    r_vals = np.empty((nstat, n_tsteps_mcmc), dtype=float)
    #  ^  ^ signature: a[i_iter, i_t] = R(t)   # i_iter = MCMC iteration

    np.random.seed(seed)

    # Set first time step values to zero (when there's no data on the left)
    if i_t0 > 0:
        r_vals[:, 0] = 0.

    # Allocate array for the effective (fwd or bkd) generation time
    omega_1d = np.empty(tg_max, dtype=float)

    # Time loop
    # ---------
    for i_t in range(i_t0, n_tsteps):
        data_i = data[i_t]   # Unchecked: assumes len(data) == n_tsteps

        # --- Make sure that tg is a backward distribution
        if tg_is_forward:
            # --- Take backwards (i_t - s) and renormalize (by sumomega)
            omega_1d[0] = 0.
            for s in range(1, tg_max):
                omega_1d[s] = omega[i_t - s, s]
            omega_1d /= omega_1d[1:].sum()
        else:
            omega_1d[:] = omega[i_t, :]

        # --- Calculates the average past incidence, weighted by tg
        past_c = 0.
        for s in range(1, min(i_t + 1, tg_max)):
            # s = generation time value
            # past_c += data[i_t - s] * omega[i_t, s]
            past_c += data[i_t - s] * omega_1d[s]

        # --- Initializes the chain
        r_curr = 1.
        loglike_curr = _numba_poisson_loglikelihood(
            r_curr, data_i, past_c
        )

        # MCMC loop
        # ---------

        # --- Transient chain
        for i_iter in range(ntrans):
            # --- Samples a candidate reproduction number
            r_cand = -1
            while r_cand < 0.:
                r_cand = np.random.normal(
                    loc=r_curr, scale=sigma * (1 + data_i))

            loglike_cand = _numba_poisson_loglikelihood(
                r_cand, data_i, past_c
            )

            # --- Chooses between current and candidate
            if np.random.random() < np.exp(loglike_cand - loglike_curr):
                r_curr = r_cand
                loglike_curr = loglike_cand

        # --- Stationary chain
        for i_iter in range(nstat):
            # --- Samples a candidate reproduction number
            r_cand = -1
            while r_cand < 0.:
                r_cand = np.random.normal(
                    loc=r_curr, scale=sigma * (1 + data_i))

            loglike_cand = _numba_poisson_loglikelihood(
                r_cand, data_i, past_c
            )

            # --- Chooses between current and candidate
            if np.random.random() < np.exp(loglike_cand - loglike_curr):
                r_curr = r_cand
                loglike_curr = loglike_cand

            # --- Stores current
            r_vals[i_iter, i_t - i_t0] = r_curr

    return r_vals

