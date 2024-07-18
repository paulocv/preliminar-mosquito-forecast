"""
Hub module for methods to synthesise future data from previous one.
"""

import numba as nb
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import time
import warnings

from collections import defaultdict
from colorama import Fore, Style

from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ
from rtrend_tools.forecast_structs import TgData
from rtrend_tools.rt_estimation import McmcRtEnsemble


# ----------------------------------------------------------------------------------------------------------------------
# FREQUENT AUXILIARY METHODS
# ----------------------------------------------------------------------------------------------------------------------

def make_tg_gamma_vector(shape, rate, tmax):
    """
    Prepare a vector with tg(s), where s is the integer time (in days) since infection.
    The vector is normalized but the FIRST ENTRY IS NOT REGARDED for the normalization.
    """

    def tg_dist_unnorm(t):
        return scipy.stats.gamma.pdf(t, a=shape, scale=1. / rate)

    st_array = np.arange(tmax, dtype=int)  # Retrograde time vector
    tg_array = tg_dist_unnorm(st_array)  #
    tg_array /= tg_array[1:].sum()  # Normalize EXCLUDING THE FIRST ELEMENT (which remains in there)

    return tg_array


def make_tg_gamma_matrix(shape_arr, rate_arr, tmax):
    """
    Create a time series of gamma distributions, assuming a time series of parameters shape and rate.
    These arrays must have the same numpy shape.
    """
    if shape_arr.shape != rate_arr.shape:
        raise ValueError(f"Hey, shape_arr {shape_arr.shape} and rate_arr {rate_arr.shape} must have same numpy shape.")

    out = np.empty((shape_arr.shape[0], tmax), dtype=float)
    st_array = np.arange(tmax, dtype=int)

    for i_t, (shape, rate) in enumerate(zip(shape_arr, rate_arr)):
        tg_array = scipy.stats.gamma.pdf(st_array, a=shape, scale=1. / rate)  #
        out[i_t] = tg_array / tg_array[1:].sum()  # Normalize EXCLUDING THE FIRST ELEMENT (which remains in there)

    return out


def _get_arg_or_kwarg(args, kwargs, i_arg, key):
    """
    Use this function to get a required argument that can also be passed as keyword argument
    (using *args and **kwargs)
    """

    try:
        val = args[i_arg]
    except IndexError:
        try:
            val = kwargs[key]
        except KeyError:
            raise TypeError(f"Hey, argument '{key}' is required.")

    return val


def apply_ramp_to(arr, k_start, k_end, ndays_fore):
    # --- Apply the slope
    ramp = np.linspace(k_start, k_end, ndays_fore)
    return np.apply_along_axis(lambda x: x * ramp, axis=1, arr=arr)


def _k_start_f_default(x):
    return 0.93


def _k_end_f_default(x):
    # return -0.156 * x + 0.9375  # 1.6 to 1.2  |  1.1 to 0.9
    return -0.520 * x + 1.46  # 1.6 to 1.2  |  1.0 to 1.0


def apply_dynamic_ramp_to(arr, ndays_fore, k_start_f=_k_start_f_default, k_end_f=_k_end_f_default,
                          x_array=None):
    raise NotImplementedError
#     """
#     Dynamic ramp, where k_start and k_end depend on an array of arguments.
#     This array can be informed as x_array, expected to have the same number of samples as arr. If not informed, the
#     first time point of each sample in arr (i.e., arr[:,0]) is used.
#
#     Parameters k_start_f and k_end_f are callables (scalar to scalar, array to array) that return k_start and k_end for
#     each value in x_array.
#     """
#     if x_array is None:
#         x_array = arr[:, 0]
#
#     if x_array.shape[0] != arr.shape[0]:
#         raise ValueError("Hey, x_array must have the same size as the first dimension of arr.")
#
#     # --- Apply ramp
#     ramp = np.linspace(k_start_f(x_array), k_end_f(x_array), ndays_fore).T
#
#     return ramp * arr


def _k_linear_func(x, r1, r2):
    return (r2/2 - r1) * (x - 1) + r1


def apply_linear_dynamic_ramp_to(arr, ndays_fore, r1_start, r1_end, r2_start, r2_end, x_array=None, **kwargs):
    """
    Dynamic ramp, where k_start and k_end depend on an array of arguments.
    This array can be informed as x_array, expected to have the same number of samples as arr. If not informed, the
    first time point of each sample in arr (i.e., arr[:,0]) is used.

    --- For each of the positions, start and end, it defines a k-coeffient function:
    k(R) = a * R + b, where
        a = r2 / 2 - r1
        b = r1
    And thus:
        R'(1) = r1
        R'(2) = r2


    """
    # --- Argument (x-array) array parsing
    if x_array is None:
        x_array = arr[:, 0]

    if x_array.shape[0] != arr.shape[0]:
        raise ValueError("Hey, x_array must have the same size as the first dimension of arr.")

    # --- Apply ramp
    ramp = np.linspace(_k_linear_func(x_array, r1_start, r2_start),
                       _k_linear_func(x_array, r1_end, r2_end),
                       ndays_fore).T
    return ramp * arr


# ----------------------------------------------------------------------------------------------------------------------
# R(t) SYNTHESIS: API METHODS
# ----------------------------------------------------------------------------------------------------------------------

def flat_mean(rtm: McmcRtEnsemble, ct_past, ndays_fore, **kwargs):
    """
    Synthesizes R(t) by taking the mean R(t) of the last ndays.
    """

    raise NotImplementedError()


def random_normal_synth(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Sample R(t) from a normal distribution.
    """
    center = kwargs.get("center", 1.0)
    sigma = kwargs.get("sigma", 0.05)
    seed = kwargs.get("seed", None)
    num_samples = kwargs.get("num_samples", None)
    out = kwargs.get("out", None)

    if num_samples is None:
        num_samples = rtm.get_array().shape[0]

    if out is None:
        out = np.empty((num_samples, ndays_fore), dtype=float)

    # Sample normal values
    rng = np.random.default_rng(seed)
    vals = rng.normal(center, sigma, num_samples)

    # Remove negative R(t) values
    vals[vals < 0] = 0

    # Repeat values over time
    for i in range(ndays_fore):
        out[:, i] = vals

    return out


def sorted_flat_mean_ensemble(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Synth a sample of flat R(t) curves from sorted ensembles of R(t) in the past (sorted over iterations in each t).
    """
    ndays_past = _get_arg_or_kwarg(args, kwargs, 0, "ndays_past")

    # --- Keyword arguments parsing
    q_low = kwargs.get("q_low", 0.25)
    q_hig = kwargs.get("q_hig", 0.75)
    r_max = kwargs.get("r_max", None)
    out = kwargs.get("out", None)

    # --- Filter by quantiles
    i_low, i_hig = [round(rtm.iters * q) for q in (q_low, q_hig)]  # Indexes of the lower and upper quantiles
    filt = rtm.get_sortd()[i_low:i_hig, -ndays_past:]

    mean_vals = filt.mean(axis=1)

    # -()- Filter by maximum R value.
    if r_max is not None:
        # # -()- Generic mask
        # mean_vals = mean_vals[mean_vals < r_max]
        # -()- Assuming sorted
        mean_vals = mean_vals[:np.searchsorted(mean_vals, r_max, side="right")]

        if mean_vals.shape[0] == 0:
            warnings.warn(f"\nHey, all mean values were above r_max = {r_max}..\n")

    # --- Produce the R(t) ensemble from the means of the past series.
    if out is None:
        out = np.empty((mean_vals.shape[0], ndays_fore), dtype=float)

    for i in range(ndays_fore):
        out[:, i] = mean_vals

    return out


def sorted_bent_mean_ensemble(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):

    k_start = kwargs.get("k_start", 1.0)
    k_end = kwargs.get("k_end", 0.8)

    # --- Get the flat version
    out = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)

    if out.shape[0] != 0:  # Ensemble can't be empty
        # --- Apply the slope
        out = apply_ramp_to(out, k_start, k_end, ndays_fore)

    return out


def sorted_bent_ensemble_valbased(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Similar to sorted_bent_mean_ensemble, but take the quantiles based on maximum and/or minimum values instead.
    """
    # First define the quantiles to get the whole sample
    kwargs["q_low"] = 0.0
    kwargs["q_hig"] = 1.0
    k_start = kwargs.get("k_start", 1.0)
    k_end = kwargs.get("k_end", 0.8)
    rmean_top = kwargs.get("rmean_top", 1.2)  # Maximum value of R (from mean ensemble) to base the range on
    max_width = kwargs.get("max_width", 0.2)  # Maximmum width in the range, below rmean_top

    # Take the full flat ensemble
    out = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)

    # Select based on rmean_top and max_width
    i_top = np.searchsorted(out[:, 0], rmean_top)
    int_width = min(round(rtm.iters * max_width), i_top)  # Width in integer indexes. Capped by the sample size.
    out = out[i_top - int_width:i_top]  # Slices the array

    # --- Apply the slope
    if out.shape[0] != 0:  # Ensemble can't be empty
        out = apply_ramp_to(out, k_start, k_end, ndays_fore)

    return out


def sorted_dynamic_ramp(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """
    Applies an R(t) ramp whose slope depends on the average value.
    """
    # k_start = kwargs.get("k_start", 1.0)
    # k_end = kwargs.get("k_end", 0.8)

    # --- Get the flat version
    out = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)

    if out.shape[0] != 0:  # Ensemble can't be empty
        # --- Apply the slope
        # out = apply_ramp_to(out, k_start, k_end, ndays_fore)
        out = apply_linear_dynamic_ramp_to(out, ndays_fore=ndays_fore, **kwargs)

    return out


def median_linear_regression(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """Extrapolates from a linear regression with the past median. Creates an ensemble around it."""

    # Argument parsing
    ndays_past = _get_arg_or_kwarg(args, kwargs, 0, "ndays_past")
    # q_low = kwargs.get("q_low", 0.25)
    # q_hig = kwargs.get("q_hig", 0.75)
    # r_max = kwargs.get("r_max", None)
    out = kwargs.get("out", None)

    stretch = (0.9, 1.1)  # Values to extend the linear fit
    nsamples = 500  # Number of stretched samples to create

    # --- Produce the R(t) ensemble from the means of the past series.
    if out is None:
        out = np.empty((nsamples, ndays_fore), dtype=float)

    # Median and linear regression
    med_series = rtm.df.median()[-ndays_past:]
    lin_coefs = np.polyfit(med_series.index, med_series.values, deg=1)
    lin_f = np.poly1d(lin_coefs)
    t_fore = np.arange(med_series.index[-1] + 1, med_series.index[-1] + 1 + ndays_fore)
    out_baseline = lin_f(t_fore)  # Applies the fit for future labels to get a baseline synthetic R(t)

    # Expansion (stretch) around median to obtain an ensemble
    starts = np.linspace(stretch[0] * out_baseline[0], stretch[1] * out_baseline[0], nsamples)
    stops = np.linspace(stretch[0] * out_baseline[-1], stretch[1] * out_baseline[-1], nsamples)
    out[:] = np.linspace(starts, stops, ndays_fore).T

    return out


def synth_file_ensemble(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """Employs an R(t) ensemble loaded from a file."""
    fname = kwargs["fname"]
    i_pres = kwargs["i_pres"]
    q_low = kwargs.get("q_low", 0.25)
    q_hig = kwargs.get("q_hig", 0.75)
    is_sorted = kwargs.get("is_sorted", True)

    # Load file and crop forecast region
    rt_array = np.loadtxt(fname) if not isinstance(fname, np.ndarray) else fname
    # ^ ^ Signature: a[i_sample, i_date]

    # Crop the region from the file ensemble
    i_days = np.arange(i_pres, i_pres+ndays_fore) % rt_array.shape[1]  # Cycles through the file.
    rt_fore = rt_array[:, i_days]

    if not is_sorted:  # Optional sorting
        rt_fore = np.sort(rt_fore, axis=0)

    # Filter quantiles
    i_low, i_hig = [round(rt_fore.shape[0] * q) for q in (q_low, q_hig)]
    return rt_fore[i_low:i_hig]


def synth_file_as_slope(rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """Use the near-past R(t) average value, then slopes (ramp) using past-season R(t) curves."""
    out = kwargs.get("out", None)
    i_pres = kwargs["i_pres"]
    ndays_past = kwargs["ndays_past"]

    # --- Request extra days from the file ensemble, to make a near-past average
    kwargs["i_pres"] = i_pres - ndays_past
    ndays_fore_file = ndays_fore + ndays_past

    # ---()--- Get both file ensemble and near-past average ensemble
    # COMMENT THIS TO USE OLD VERSION!!
    rt_fore_avg = sorted_flat_mean_ensemble(rtm, ct_past, ndays_fore, *args, **kwargs)
    rt_fore_file = synth_file_ensemble(rtm, ct_past, ndays_fore_file, *args, **kwargs)
    # ^ ^ signature: a[i_sample, i_date]

    # Ensemble sizes check
    if rt_fore_file.shape[0] != rt_fore_avg.shape[0]:
        print(Fore.YELLOW +
              f"Hey, the numbers of samples from the file ensemble ({rt_fore_file.shape[0]}) and the near-past"
              f" average ({rt_fore_avg.shape[0]}) do not match. This can be solved with sampling, but is currently "
              f"not implemented." + Style.RESET_ALL)
        return np.empty((0, 0))  # Tells method to skip this

    if out is None:
        out = np.zeros_like(rt_fore_avg)

    # ------
    # Calculate the average of the latest file ensemble days
    rt_file_near_avg = rt_fore_file[:, 0:ndays_past].mean(axis=1)[:, np.newaxis]  # Take average of the past part

    # Perform the combination between file and MCMC ensembles
    # ---()--- New version – takes an average over ndays_past last days as the slope denominator
    slopes = rt_fore_file[:, ndays_past:] / rt_file_near_avg  # Divide each sample by first pt. Assumes non-zero.
    out[:] = rt_fore_avg * slopes

    # # ---()---  Old version - takes only the first fore day
    # slopes = rt_fore_file[:, :] / rt_fore_file[:, 0][:, np.newaxis]  # Divide each sample by first pt. Assumes non-zero.
    # out[:] = rt_fore_avg * slopes

    return out


# ----------------------------------------------------------------------------------------------------------------------
# Tg(t) SYNTHESIS
# ----------------------------------------------------------------------------------------------------------------------


# Gamma TG distribution - Conversions between shape/rate and mean/std (u/v)
# -------------------------------------------------------------------------
def _gamma_sr_to_u(s, r):  # From shape/rate to mean
    return s / r


def _gamma_sr_to_v(s, r):  # From shape/rate to standard deviation
    return np.sqrt(s) / r


def _gamma_uv_to_s(mu, std):  # From mean/std to shape
    return mu**2 / std**2


def _gamma_uv_to_r(mu, std):
    return mu / std**2


# -------------------------
# -------------------------


def tg_synth_last_value(tg: TgData, rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """Simply takes the value in the last step."""
    return np.repeat(tg.df_roi["Shape"].iloc[-1], ndays_fore), np.repeat(tg.df_roi["Rate"].iloc[-1], ndays_fore)


def tg_synth_near_past_avg(tg: TgData, rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """Takes the average over the last TG values in the mean-std space."""
    # --- Take near-past values
    ndays_past = kwargs["ndays_past"]
    shape_past_array = tg.df_roi["Shape"].iloc[-ndays_past:].values
    rate_past_array  = tg.df_roi["Rate"].iloc[-ndays_past:].values

    # --- Convert to mean/std, then calculate the average
    mean_past_avg = _gamma_sr_to_u(shape_past_array, rate_past_array).mean()
    std_past_avg  = _gamma_sr_to_v(shape_past_array, rate_past_array).mean()

    # --- Back to shape/rate
    shape_past_avg = _gamma_uv_to_s(mean_past_avg, std_past_avg)
    rate_past_avg  = _gamma_uv_to_r(mean_past_avg, std_past_avg)

    return np.repeat(shape_past_avg, ndays_fore), np.repeat(rate_past_avg, ndays_fore)


def tg_synth_file_series(tg: TgData, rtm: McmcRtEnsemble, ct_past, ndays_fore, *args, **kwargs):
    """Grabs the generation time parameters from a file."""
    tg_file = kwargs["tg_file"]
    day_pres: pd.Timestamp = kwargs["day_pres"]

    # Read file and attribute dates
    if not isinstance(tg_file, pd.DataFrame):
        f_tg_df = pd.read_csv(tg_file, index_col=0)
        # f_tg_df
    else:
        raise NotImplementedError

    # Get the days to be forecast.
    days_fore = pd.date_range(day_pres, periods=ndays_fore, freq="D")
    idx = (days_fore.day_of_year - 1) % f_tg_df.shape[0] + 1  # Colapses to valid 1-year values.
    loc_df = f_tg_df.loc[idx]  # Takes Tg info from the selected days of the year

    return loc_df["Shape"], loc_df["Rate"]


# ---------------
# DICTIONARY OF SYNTH METHODS that can be used as default
SYNTH_METHOD = {
    "static_ramp": sorted_bent_mean_ensemble,
    "linear_reg": median_linear_regression,
    "file_ens": synth_file_ensemble,
    "file_as_slope": synth_file_as_slope,
    "rnd_normal": random_normal_synth,
}

TG_SYNTH_METHOD = {
    "last_value": tg_synth_last_value,
    "near_past_avg": tg_synth_near_past_avg,
    "file_series": tg_synth_file_series,
}

# ----------------------------------------------------------------------------------------------------------------------
# C(t) RECONSTRUCTION
# ----------------------------------------------------------------------------------------------------------------------


def reconstruct_ct(ct_past, rt_fore, tg_array, tg_max=None, ct_fore=None, seed=None, **kwargs):
    """
    This function will later be the wrapper for numbaed functions. It should be able to handle both single R(t) series
    and ensembles of these as well.


    (draft, under construction)
    NOTICE: the calculation is recursive (C(t) depends on C(t - s)), thus cannot be vectorized.
    ------
    ct_past :
        Vector of number of cases in the past.
    rt_fore : Union(np.ndarray, pd.Series, pd.DataFrame)
        A time series with R(t) in future steps. Its size is assumed as steps_fut.
        For now, I expect a pandas Series. I can review this as needed though. E.g. accept 2d arrays.
        The size of the forecast is assumed to be the size of this vector.
    tg_array :
        Vector with Tg(s), where s is time since infection.
    ct_fore :
        c_futu. Contains the forecast.
    tg_max :
        Truncation value (one past) for generation time.
    """
    num_steps_fore = rt_fore.shape[-1]

    # Input types interpretation
    # --------------------------

    # --- R(t) input
    run_mult = False  # Whether to run the reconstruction over multiple R(t) series.
    if isinstance(rt_fore, np.ndarray):  # Numpy array
        if rt_fore.ndim == 2:
            run_mult = True
        elif rt_fore.ndim > 2:
            raise ValueError(f"Hey, input R(t) array rt_fore must either be 1D or 2D (ndim = {rt_fore.ndim}).")

    elif isinstance(rt_fore, pd.Series):  # Pandas 1D series
        rt_fore = rt_fore.to_numpy()

    elif isinstance(rt_fore, pd.DataFrame):  # Pandas 2D data frame
        rt_fore = rt_fore.to_numpy()
        run_mult = True

    else:
        raise TypeError(f"Hey, invalid type for input R(t) rt_fore: {type(rt_fore)}")

    # --- C(t) past array
    if isinstance(ct_past, pd.Series):
        ct_past = ct_past.to_numpy()

    # --- Tg(s) array
    if isinstance(tg_array, pd.Series):
        tg_array = tg_array.to_numpy()

    # Optional arguments handling
    # ---------------------------

    if tg_max is None:
        tg_max = tg_array.shape[0]

    # Output array
    if ct_fore is None:
        ct_fore = np.empty_like(rt_fore, dtype=int)  # Agnostic to 1D or 2D

    # Time-based seed
    if seed is None:
        seed = round(1000 * time.time())

    # Dispatch and run the reconstruction
    # -----------------------------------
    if run_mult:
        return _reconstruct_ct_multiple(ct_past, rt_fore, tg_array, tg_max, ct_fore,
                                        num_steps_fore, seed)
    else:
        return _reconstruct_ct_single(ct_past, rt_fore, tg_array, tg_max, ct_fore,
                                      num_steps_fore, seed)


@nb.njit
def _reconstruct_ct_single(ct_past, rt_fore, tg_dist, tg_max, ct_fore, num_steps_fore, seed):

    np.random.seed(seed)  # Seeds the numba or numpy generator

    # Main loop over future steps
    for i_t_fut in range(num_steps_fore):  # Loop over future steps
        lamb = 0.  # Sum of generated cases from past cases
        # r_curr = rt_fore.iloc[i_t_fut]  # Pandas
        r_curr = rt_fore[i_t_fut]

        # Future series chunk
        for st in range(1, i_t_fut + 1):
            lamb += r_curr * tg_dist[st] * ct_fore[i_t_fut - st]

        # Past series chunk
        for st in range(i_t_fut + 1, tg_max):
            lamb += r_curr * tg_dist[st] * ct_past[-(st - i_t_fut)]

        # Poisson number
        ct_fore[i_t_fut] = np.random.poisson(lamb)

    return ct_fore


@nb.njit
def _reconstruct_ct_multiple(ct_past, rt_fore_2d, tg_dist, tg_max, ct_fore_2d, num_steps_fore, seed):

    np.random.seed(seed)  # Seeds the numba or numpy generator

    # Main loop over R(t) samples
    for rt_fore, ct_fore in zip(rt_fore_2d, ct_fore_2d):

        # Loop over future steps
        for i_t_fut in range(num_steps_fore):  # Loop over future steps
            lamb = 0.  # Sum of generated cases from past cases
            # r_curr = rt_fore.iloc[i_t_fut]  # Pandas
            r_curr = rt_fore[i_t_fut]

            # Future series chunk
            for st in range(1, i_t_fut + 1):
                lamb += r_curr * tg_dist[st] * ct_fore[i_t_fut - st]

            # Past series chunk
            for st in range(i_t_fut + 1, tg_max):
                lamb += r_curr * tg_dist[st] * ct_past[-(st - i_t_fut)]

            # Poisson number
            ct_fore[i_t_fut] = np.random.poisson(lamb)

    return ct_fore_2d


def reconstruct_ct_tgtable(ct_past, rt_fore, tg: TgData, ct_fore=None, seed=None, tg_is_forward=True, **kwargs):
    """Reconstructs a time series of cases using a time-dependent generation time distribution, given as 2d arrays."""

    # Optional arguments handling
    # ---------------------------
    num_steps_fore = rt_fore.shape[-1]
    ndays_roi = ct_past.shape[-1]

    # Output array
    if ct_fore is None:
        ct_fore = np.empty_like(rt_fore, dtype=int)  # Agnostic to 1D or 2D

    # Time-based seed
    if seed is None:
        seed = round(1000 * time.time())

    # Execution
    # ---------
    return _reconstruct_ct_tgtable_multiple(
        ct_past, rt_fore, tg.past_2d_array, tg.fore_2d_array, tg.max, ct_fore,
        num_steps_fore, ndays_roi, tg_is_forward, seed
    )


@nb.njit
def _reconstruct_ct_tgtable_multiple(
        ct_past, rt_fore_2d, tg_2d_past, tg_2d_fore, tg_max, ct_fore_2d,
        num_steps_fore, ndays_roi, tg_is_forward, seed
):
    np.random.seed(seed)  # Seeds the numba or numpy generator

    tg_1d = np.empty(tg_max, dtype=float)  # Allocate the effective generation time
    tg_1d[0] = 0.

    # Main loop over R(t) samples
    for rt_fore, ct_fore in zip(rt_fore_2d, ct_fore_2d):
        # Loop over future steps

        for i_t_fut in range(num_steps_fore):  # Loop over future steps
            # Convert generation time to backward
            if tg_is_forward:
                # Convert to backward
                for s in range(1, i_t_fut + 1):
                    tg_1d[s] = tg_2d_fore[i_t_fut - s, s]
                for s in range(i_t_fut + 1, tg_max):
                    tg_1d[s] = tg_2d_past[-(s - i_t_fut), s]
                # Renormalize
                tg_1d[:] /= tg_1d.sum()
            else:
                # Assume backwards, use tg future only. Assume it's normalized.
                tg_1d[1:] = tg_2d_fore[i_t_fut, 1:]

            lamb = 0.  # Sum of generated cases from past cases
            # r_curr = rt_fore.iloc[i_t_fut]  # Pandas
            r_curr = rt_fore[i_t_fut]

            # Future series chunk
            for st in range(1, i_t_fut + 1):
                # lamb += r_curr * tg_2d_fore[i_t_fut - st, st] * ct_fore[i_t_fut - st]
                lamb += r_curr * tg_1d[st] * ct_fore[i_t_fut - st]

            # Past series chunk
            for st in range(i_t_fut + 1, tg_max):
                # lamb += r_curr * tg_2d_past[-(st - i_t_fut), st] * ct_past[-(st - i_t_fut)]
                lamb += r_curr * tg_1d[st] * ct_past[-(st - i_t_fut)]

            # Poisson number
            # NOTE: a negative or unreasonable value here may be a day_pres w/out enough past data.
            #   This is currently not handled outside (2023-06-16).

            # # --- BREAKPOINT!!!
            # if lamb < 0:
            #     ct_fore[i_t_fut] = 0
            #     print("NEGATIVE LAMBDA: ", "")
            #     print(lamb)
            #     continue
            # ---
            ct_fore[i_t_fut] = np.random.poisson(lamb)

    return ct_fore_2d


# @nb.njit  # Not worth the compilation time at each parallel session.
def sample_series_with_replacement(ct_fore_2d: np.ndarray, nsamples_us: int, seed):
    """Randomly sample the ensemble of forecasted values, with the aim of feeding the US time series."""

    nsamples_state, nsteps = ct_fore_2d.shape
    np.random.seed(seed)  # Seeds numpy or numba generator.

    return ct_fore_2d[np.random.randint(0, nsamples_state, size=nsamples_us)]


def filter_ct_trajectories_by_max(ct_fore2d: np.ndarray, ct_max, min_required):
    """Returns a view of the forecast trajectories array excluding those that
    exceed a certain value.

    If too little trajectories are left (not enough for a good quantile calculation),
    raises a warning and returns the unfiltered data.
    """
    mask: np.ndarray = (ct_fore2d.max(axis=1) <= ct_max)

    # Check if the minimum number of passed values was achieved
    if mask.sum() < min_required:
        msg = f"A low number of c(t) trajectories ({mask.sum()}) passed the maximum value filter." \
              f" The filter was bypassed."
        warnings.warn(Fore.YELLOW + msg + Style.RESET_ALL)

        # # -()- # Returns unfiltered data
        # return ct_fore2d

        # -()- Returns NaN array
        ct_fore2d[:] = np.nan
        return ct_fore2d

    return ct_fore2d[mask, :]


def calc_tseries_ensemble_quantiles(ct_fore2d, quantiles_seq=None):
    """
    Calculate a sequence of quantiles for each point of a time series ensemble.
    Input signature: ct_fore2d[i_sample, i_t]

    If a custom sequence of quantiles is not informed, uses the default from CDC.
    """

    if quantiles_seq is None:
        quantiles_seq = CDC_QUANTILES_SEQ

    return np.quantile(ct_fore2d, quantiles_seq, axis=0)
