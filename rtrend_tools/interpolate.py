"""
Module for INTERPOLATION and AGGREGATION of weekly hospitalization data into daily data.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.interpolate import UnivariateSpline

WEEKLEN = 7  # Number of steps (days) in a week.
NEGATIVE_VAL_TOL = -15  # Tolerance for negative values at interpolation.


# ----------------------------------------------------------------------------------------------------------------------

def weekly_to_daily_spline(t_weekly: np.ndarray, data_weekly: np.ndarray, t_daily=None, rel_smooth_fac=0.01,
                           spline_degree=3, return_complete=False, **kwargs):
    """
    TODO WRITE DOCS

    Assumes that each weekly time stamp t represents the past seven days, from t-1 to t-7 (WEEKLEN).
    If not informed, the daily t array is constructed with the hypothesis above.
    """

    # PREAMBLE
    # -------------------
    num_week_points = data_weekly.shape[0]

    if t_weekly.shape[0] != num_week_points:
        raise ValueError(f"Hey, number of points in data ({num_week_points}) does not match the number of points in "
                         f"weekly time array ({t_weekly.shape[0]}).")

    if t_daily is None:  # Constructs the array of consecutive days, from day 0 to last day in data.
        t_daily = np.arange(t_weekly[0] - WEEKLEN, t_weekly[-1], 1, dtype=int)

    if isinstance(data_weekly, pd.Series):
        data_weekly = data_weekly.array

    # EXECUTION
    # -----------------

    # --- Cumulative incidence
    t_weekly = np.insert(t_weekly, 0, t_weekly[0] - WEEKLEN)  # Extra point to complete the spline
    data_weekly = np.insert(data_weekly, 0, data_weekly[0])
    csum = np.cumsum(data_weekly)

    # --- Spline Interpolation
    spline = UnivariateSpline(t_weekly, csum, s=num_week_points * rel_smooth_fac, k=spline_degree,
                              ext="const")
    csum_daily = spline(t_daily)

    # print(spline.get_coeffs())  # WATCHPOINT
    # print(csum)
    # print((t_weekly[1:] - t_weekly[:-1]).argmin())

    # --- Reconstruction of the daily data from cumulative
    # float_data_daily = np.empty(csum_daily.shape[0] - 1, dtype=float)
    float_data_daily = np.empty(csum_daily.shape[0], dtype=float)
    float_data_daily[0] = csum_daily[1] - csum_daily[0]  # Repeat next value for reasonability. Fix later.
    float_data_daily[1:] = csum_daily[1:] - csum_daily[:-1]  # Has negative values due to "local bellies"

    data_daily = np.round(float_data_daily).astype(int)

    # --- Simple negative values handling
    neg_its = list()  # Stores the negative indexes
    for i_t, val in enumerate(data_daily):
        if val >= 0:
            continue

        if val <= NEGATIVE_VAL_TOL:  # Value too negative, could mean trouble.
            sys.stderr.write("Negative value over threshold: "
                             f"ct_past[{i_t}] = {val}\n")

        # Register
        neg_its.append(t_daily[i_t])

        # Set to zero and "blame" next step.
        data_daily[i_t] = 0
        try:
            data_daily[i_t + 1] += val  # Note: val is negative, so this is a subtraction
        except IndexError:  # Just ignore if time step is the last one.
            pass

    if return_complete:
        return t_daily, data_daily, spline, float_data_daily
    else:
        # return t_daily, data_daily
        return t_daily, data_daily, neg_its  # TODO: JUST TO DEBUG


def weekly_to_daiy_experimental(t_weekly, data_weekly):
    """Sandbox function for testing some interpolation methods"""
    # PREAMBLE
    # -------------------
    num_week_points = data_weekly.shape[0]

    if t_weekly.shape[0] != num_week_points:
        raise ValueError(f"Hey, number of points in data ({num_week_points}) does not match the number of points in "
                         f"weekly time array ({t_weekly.shape[0]}).")

    # t_daily = np.arange(t_weekly[0] - WEEKLEN, t_weekly[-1], 1, dtype=int)
    t_daily = np.arange(t_weekly[0] - WEEKLEN, t_weekly[-1] + WEEKLEN, 1, dtype=int)  # Good for uniform interpolation

    if isinstance(data_weekly, pd.Series):
        data_weekly = data_weekly.array

    t_weekly = np.insert(t_weekly, 0, t_weekly[0] - WEEKLEN)  # Extra point to complete the spline
    data_weekly = np.insert(data_weekly, 0, data_weekly[0])

    # # # -()- Direct spline interpolation ---
    # print(f"WATCH NUM WEEK = {num_week_points}")
    # spline = UnivariateSpline(
    #     t_weekly, data_weekly, s=0, k=1, ext="const"  # No smoothing
    #     # t_weekly, data_weekly, s=0.5E4 * num_week_points, k=1, ext="const"  # Smoothing
    # )
    # float_data_daily = spline(t_daily)

    # -()- Uniform interpolation ---
    # WARNING: this method is simple but will NOT account for MISSING WEEKS!!!
    float_data_daily = np.repeat(data_weekly, WEEKLEN).astype(float)

    # # --- ADDONS
    # -()- Divide by week length (7)
    # # float_data_daily /= WEEKLEN

    # # -()- Add Poisson sampling with down-up scaling
    # downup_scale = 20  # Downscales before poisson, upscales back again
    # rng = np.random.default_rng(100)
    # float_data_daily /= downup_scale  # Downscale
    # float_data_daily = rng.poisson(float_data_daily)  # Poisson noise
    # float_data_daily *= downup_scale  # Upscale

    # # -()- Replace back the original data in week reference dates
    # # Using pandas series just to simplify
    # daily_sr = pd.Series(float_data_daily, index=t_daily)
    # daily_sr.loc[t_weekly[:-1]] = data_weekly[:-1]
    # #  ^ ^ Last day not present in daily data, so it's excluded

    # --- TURN INTO INTEGERS
    data_daily = np.round(float_data_daily).astype(int)

    neg_its = list()

    return t_daily, data_daily, neg_its


def weekly_to_daily_uniform(t_weekly, data_weekly):
    """
    Uniformly distributes each value in data through the past 7 days. (WEEKLEN)
    Array t_daily is constructed on the fly.
    """

    # PREAMBLE
    # -------------------

    num_week_points = data_weekly.shape[0]

    if t_weekly.shape[0] != num_week_points:
        raise ValueError(f"Hey, number of points in data ({num_week_points}) does not match the number of points in "
                         f"weekly time array ({t_weekly.shape[0]}).")

    if isinstance(data_weekly, pd.Series):
        data_weekly = data_weekly.array

    # EXECUTION
    # -------------------

    # TODO: THERE should be a way to do this faster with numpy.tile or repeat. May speedup plots.

    t_daily = np.empty(WEEKLEN * num_week_points, dtype=int)
    data_daily = np.empty(WEEKLEN * num_week_points, dtype=float)

    # for i_data, t in enumerate(t_weekly[1:]):
    for i, (day, val) in enumerate(zip(t_weekly, data_weekly)):

        d_value = val / WEEKLEN  # Current weekly hospitalizations

        start, end = WEEKLEN * i, WEEKLEN * (i + 1)
        t_daily[start:end] = np.arange(start, end)
        data_daily[start:end] = d_value

    return t_daily, data_daily


# ----------------------------------------------------------------------------------------------------------------------

def daily_to_weekly(data_daily):
    """
    CURRENTLY: discards last days that do not fit in a new week.
    SIMPLE METHOD: assumes that there are no missing days. That's why it takes only one array as input.
    TODO: FINISH THIS METHOD

    ct_past: This works both for a single time series (a[i_t]) and an ensemble of time series (a[samp, i_t]),
    although be watchful for performance flaws with large samples (a specifically designed function might work better).

    WORKS WITH NUMPY ARRAY, not pandas series YET.
    """
    num_days = data_daily.shape[-1]
    num_weeks = num_days // WEEKLEN

    # TODO: CHECK SHAPE EQUALITY BETWEEN ARRAYS.

    data_weekly = np.zeros((*data_daily.shape[:-1], num_weeks), int)

    for i_day, val in enumerate(data_daily.T):

        i_week = i_day // WEEKLEN
        if i_week >= num_weeks:  # Series is over, remaining days are discarded.
            break

        data_weekly.T[i_week] += val

    return data_weekly
