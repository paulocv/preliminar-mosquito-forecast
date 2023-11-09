"""
This code was adapted from the publicly available file at:
> https://github.com/adrian-lison/interval-scoring

Fetch date: 2023-02-02
No license was attributed to the code by the time of fetching.
"""

import numpy as np


# AUXILIARY ROUTINES
# ------------------

def _aux_get_q_edges(q_dict, q_left, q_right, alpha):
    """Select between a q_dict with quantiles or a pair of dicts of q edges
    (left-right).
    Common to interval-based scores.
    """
    if q_dict is None:  # Expects q_left and q_right to be
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(round(alpha / 2, 6))
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha/2}-quantile")

        q_right = q_dict.get(round(1 - (alpha / 2), 6))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1-(alpha/2)}-quantile"
            )
    return q_left, q_right


# USER SCORES
# ----------------------------------------------
def alpha_covered_score(
        observations,
        alpha,
        q_dict=None,
        q_left=None,
        q_right=None,
        check_consistency=True,
        return_type=None
):
    """
    Calculates the simple "coverage" score for a given alpha value (representing the 1-alpha IQR).
    For each data in the ``observations`` array and respective quantiles (q_dict), returns True if the observation
    falls inside the range and False otherwise.

    Parameters TODO write
    ----------
    observations :
    alpha :
    q_dict :
    q_left :
    q_right :
    check_consistency :
    return_type :
    """
    q_left, q_right = _aux_get_q_edges(q_dict, q_left, q_right, alpha)

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    # Calculations
    # ------------------------------------------
    result: np.ndarray = np.logical_and(observations >= q_left, observations <= q_right)

    if return_type:
        result = result.astype(return_type)

    return result


def simple_quantile_reach_score_arraybased(
        observations,
        quantiles_seq,
        q_matrix,
        check_consistency=True,
        float_eps=1E-6,
):
    """
    Calculates the Simple Quantile Reach (SQR) score for a number of different predicted intervals.
    For each data in the `observations` array and respective quantiles (q_dict), returns the
    SQR = 1 - alpha corresponding to the greatest alpha interval that contains the truth point. If
    none of the intervals contain the point, returns 1 (alpha = 0). Higher score means lower coverage.

    `arraybased`: this version receives a sequence of quantile levels and a 2D array with interval values
    instead of dictionaries.

    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    quantiles_seq : iterable
        Dictionary with predicted quantiles for all instances in `observations`.
        Either this or `q_matrix` must be informed.
    q_matrix : np.ndarray, optional.
        A 2D array with the forecast intervals, with the following signature:
        > q_matrix[q, i_t] = forecast
        Where `q` is one of the q-values of an interval. It must satisfy either
        q = alpha / 2 or q = 1 - alpha / 2 for one of the alphas in `alphas`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
    float_eps : float, optional.
        Floating point precision for float equalities that are made within this method.

    Returns
    -------
    scores : np.ndarray
        SQR for the array of observations and predictions.
    """
    quantiles_seq = np.fromiter(quantiles_seq, dtype=float)
    num_quantiles = quantiles_seq.shape[0]
    num_intervals = num_quantiles // 2  # Excludes median
    halfway_interval = (num_quantiles + 1) // 2  # Includes median

    # Consistency checks
    # ------------------
    if check_consistency:
        # Shape of indexes (quantiles_seq) and values (q_matrix)
        if q_matrix.shape[0] != quantiles_seq.shape[0]:
            raise ValueError("Hey, `quantiles_seq` must have the same length as the first "
                             "dimension of q_matrix.")

        # Monotonicity of quantile_seq values
        if np.any(np.diff(quantiles_seq) <= 0):
            raise ValueError("Hey, `quantiles_seq` must have strictly increasing values.")

        if np.any(np.diff(q_matrix, axis=0) < 0):
            raise ValueError("Hey, `q_matrix` must have increasing values"
                             " along the quantiles axis.")

        # Quantiles sequence must be symmetrical around 0.5
        addup_array = np.any(quantiles_seq[:num_intervals] + np.flip(quantiles_seq[-num_intervals:]))
        if np.any(abs(addup_array - 1.) >= float_eps):  # Finds pairs that do not add up to 1.
            raise ValueError("Hey, `quantiles_seq` is not symmetric around its halfway. It must be made "
                             "of values such as: [a, b, ..., m, ..., 1 - b, 1 - a]")

    # Calculations
    # ------------
    above_array: np.ndarray = observations >= q_matrix  # Whether obs is above each q-value
    below_array: np.ndarray = observations <= q_matrix   # Whether obs is below each q-value

    # Last quantile to be under the observation - uses sum as a shortcut
    # print(np.argmin(above_array, axis=0) + above_array.shape[0] * (np.all(above_array)))
    last_under = above_array.sum(axis=0)

    # First quantile to be over the observation
    first_over = (~below_array).sum(axis=0)

    # TODO: the correct way is to appropriately manage cases when last_under != first over.
    #   This only happens if obs and more than one quantile interval has exactly equal values.
    #   For any other case, we'll have last_under == first_over, so it's mostly ok to ignore this.

    # --- Ignoring equality cases, just take the `last_under`
    quantiles_seq_with_1 = np.append(quantiles_seq, 1.)
    q_reach = quantiles_seq_with_1[last_under]  # 1. if obs is above every interval
    sqr_score = np.abs(2 * q_reach - 1)

    return sqr_score

#
# def simple_quantile_reach_score(
#         observations,
#         alphas,
#         q_dict=None,
#         check_consistency=True,
# ):
#     """
#     Calculates the Simple Quantile Reach (SQR) score for a number of different predicted intervals.
#     For each data in the `observations` array and respective quantiles (q_dict), returns the
#     SQR = 1 - alpha corresponding to the greatest alpha interval that contains the truth point. If
#     none of the intervals contain the point, returns 1 (alpha = 0). Higher score means lower coverage.
#
#     Parameters
#     ----------
#     observations : array_like
#         Ground truth observations.
#     alphas : iterable
#         Alpha levels for (1-alpha) intervals.
#     q_dict : dict, optional
#         Dictionary with predicted quantiles for all instances in `observations`.
#     check_consistency: bool, optional
#         If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
#
#     Returns
#     -------
#     scores : np.ndarray
#         SQR for the array of observations and predictions.
#     """
#
#     raise NotImplementedError(
#         "Please use `simple_quantile_reach_score_arraybased` instead. This method was "
#         "becoming too complicated, and was not convenient for me to have it.")
#
#     # Preparations
#     # ------------
#     # print(alphas)
#     # WATCH
#     alphas = np.fromiter(alphas, dtype=float)
#     has_median = 1. in alphas  # Whether the levels contain the median (alpha = 1). FLOAT EQUALITY.
#
#     # Reconstruct the sequence of q values
#     # --------------
#     # Check if array is sorted
#     diff = np.diff(alphas)
#     if not (np.all(diff > 0) or np.all(diff < 0)):
#         raise ValueError("Hey, `alphas` array must be monotonically increasing.")
#
#     ao2 = alphas / 2
#     if has_median:
#         q_seq = np.concatenate([ao2[:-1], [0.5], 1. - np.flip(ao2[:-1])])
#         print(q_seq)
#     else:
#         print("THEN WHAT??")


# ## Interval Score
def interval_score(
    observations,
    alpha,
    q_dict=None,
    q_left=None,
    q_right=None,
    percent=False,
    check_consistency=True,
):
    """
    Compute interval scores (1) for an array of observations and predicted intervals.
    
    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be
    specified or the quantiles need to be specified via q_left and q_right.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total interval scores.
    sharpness : array_like
        Sharpness component of interval scores.
    calibration : array_like
        Calibration component of interval scores.
        
    (1) Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378.    
    """

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(round(alpha / 2, 6))
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha/2}-quantile")

        q_right = q_dict.get(round(1 - (alpha / 2), 6))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1-(alpha/2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    if percent:
        sharpness = sharpness / np.abs(observations)
        calibration = calibration / np.abs(observations)
    total = sharpness + calibration
    return total, sharpness, calibration


# ## Weighted Interval Score


def weighted_interval_score(
    observations, alphas, q_dict, weights=None, percent=False, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the WIS-score (2). A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield the double absolute percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
        
    (2) Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint arXiv:2005.12881.
    """
    if weights is None:
        weights = np.array(alphas)/2

    def weigh_scores(tuple_in, weight):
        return tuple_in[0] * weight, tuple_in[1] * weight, tuple_in[2] * weight

    interval_scores = [
        i
        for i in zip(
            *[
                weigh_scores(
                    interval_score(
                        observations,
                        alpha,
                        q_dict=q_dict,
                        percent=percent,
                        check_consistency=check_consistency,
                    ),
                    weight,
                )
                for alpha, weight in zip(alphas, weights)
            ]
        )
    ]

    total = np.sum(np.vstack(interval_scores[0]), axis=0) / sum(weights)
    sharpness = np.sum(np.vstack(interval_scores[1]), axis=0) / sum(weights)
    calibration = np.sum(np.vstack(interval_scores[2]), axis=0) / sum(weights)

    return total, sharpness, calibration


def weighted_interval_score_fast(
    observations, alphas, q_dict, weights=None, percent=False, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the WIS-score (2). A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    This is a more efficient implementation using array operations instead of repeated calls of `interval_score`.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
        
    (2) Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint arXiv:2005.12881.
    """
    if weights is None:
        weights = np.array(alphas)/2

    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        raise ValueError("Alpha values must be sorted in ascending order.")

    reversed_weights = list(reversed(weights))

    lower_quantiles = [q_dict.get(round(alpha / 2, 6)) for alpha in alphas]
    upper_quantiles = [q_dict.get(round(1. - (alpha / 2), 6)) for alpha in reversed(alphas)]

    if any(q is None for q in lower_quantiles) or any(
        q is None for q in upper_quantiles
    ):
        raise ValueError(
            f"Quantile dictionary does not include all necessary quantiles."
        )

    lower_quantiles = np.vstack(lower_quantiles)
    upper_quantiles = np.vstack(upper_quantiles)

    # Check for consistency
    if check_consistency and np.any(
        np.diff(np.vstack((lower_quantiles, upper_quantiles)), axis=0) < 0
    ):
        raise ValueError("Quantiles are not consistent.")

    lower_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    upper_q_alphas = (2 / np.array(list(reversed(alphas)))).reshape((-1, 1))

    # compute score components for all intervals
    sharpnesses = np.flip(upper_quantiles, axis=0) - lower_quantiles

    lower_calibrations = (
        np.clip(lower_quantiles - observations, a_min=0, a_max=None) * lower_q_alphas
    )
    upper_calibrations = (
        np.clip(observations - upper_quantiles, a_min=0, a_max=None) * upper_q_alphas
    )
    calibrations = lower_calibrations + np.flip(upper_calibrations, axis=0)

    # scale to percentage absolute error
    if percent:
        sharpnesses = sharpnesses / np.abs(observations)
        calibrations = calibrations / np.abs(observations)

    totals = sharpnesses + calibrations

    # weigh scores
    weights = np.array(weights).reshape((-1, 1))

    sharpnesses_weighted = sharpnesses * weights
    calibrations_weighted = calibrations * weights
    totals_weighted = totals * weights

    # normalize and aggregate all interval scores
    weights_sum = np.sum(weights)

    sharpnesses_final = np.sum(sharpnesses_weighted, axis=0) / weights_sum
    calibrations_final = np.sum(calibrations_weighted, axis=0) / weights_sum
    totals_final = np.sum(totals_weighted, axis=0) / weights_sum

    return totals_final, sharpnesses_final, calibrations_final


# Truncated log-score
def logscore_truncated(
        observations,
        quantiles_seq,
        q_matrix,
        check_consistency=True,
        float_eps=1E-6,
        trunc_point=-10,
):
    """
    TODO: NEEDS TESTING
    """
    # Simple quantile reach – Outermost IQR to miss the observation
    sqr = simple_quantile_reach_score_arraybased(
        observations, quantiles_seq, q_matrix,
        check_consistency=check_consistency, float_eps=float_eps)

    return np.maximum(np.log(sqr), trunc_point)


# ## Outside-Interval Count (A.K.A. COVERAGE)
def outside_interval(observations, lower, upper, check_consistency=True):
    """
    Indicate whether observations are outside a predicted interval for an array of observations and predicted intervals.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    lower : array_like, optional
        Predicted lower interval boundary for all instances in `observations`.
    upper : array_like, optional
        Predicted upper interval boundary for all instances in `observations`.
    check_consistency: bool, optional
        If `True`, interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    Out : array_like
        Array of zeroes (False) and ones (True) counting the number of times observations where outside the interval.
    """
    if check_consistency and np.any(lower > upper):
        raise ValueError("Lower border must be smaller than upper border.")

    return ((lower > observations) + (upper < observations)).astype(int)


# ## Interval Consistency Score


def interval_consistency_score(
    lower_old, upper_old, lower_new, upper_new, check_consistency=True
):
    """
    Compute interval consistency scores for an old and a new interval.
    
    Adapted variant of the interval score which measures the consistency of updated intervals over time.
    Ideally, updated predicted intervals would always be within the previous estimates of the interval, yielding
    a score of zero (best).
    
    Parameters
    ----------
    lower_old : array_like
        Previous lower interval boundary for all instances in `observations`.
    upper_old : array_like, optional
        Previous upper interval boundary for all instances in `observations`.
    lower_new : array_like
        New lower interval boundary for all instances in `observations`. Ideally higher than the previous boundary.
    upper_new : array_like, optional
        New upper interval boundary for all instances in `observations`. Ideally lower than the previous boundary.
    check_consistency: bool, optional
        If interval boundaries are checked for consistency. Default is `True`.
        
    Returns
    -------
    scores : array_like
        Interval consistency scores.
    """
    if check_consistency and (
        np.any(lower_old > upper_old) or np.any(lower_new > upper_new)
    ):
        raise ValueError("Left quantile must be smaller than right quantile.")

    scores = np.clip(lower_old - lower_new, a_min=0, a_max=None) + np.clip(
        upper_new - upper_old, a_min=0, a_max=None
    )
    return scores


# ===================================
# POINT FORECAST SCORES
# ===================================

# NOTICE: the letter "M"was removed from some of the accronyms because they
#   are not means, but the point value.

# ## MAE


def mae_score(observations, point_forecasts):
    """Mean absolute error of the given array"""
    return np.abs(observations - point_forecasts).mean()


def ae_score(observations, point_forecasts):
    """Absolute error for each observation/forecast pair."""
    return np.abs(observations - point_forecasts)


# ## MAPE and sMAPE


def ape_score(observations, point_forecasts):
    """Mean Average Percentage Error.
    When the observation is zero, it's defined as a Kronecker delta.
    """
    res = np.abs(point_forecasts - observations) / np.abs(observations)

    # Fix zero-observation values
    mask = (observations == 0.)
    res[mask] = (point_forecasts[mask] == 0)

    return res


def sape_score(observations, point_forecasts):
    """Symmetric Average Percentage Score
    When the observation and forecast are zero, returns zero.
    """
    res = (
        2
        * np.abs(point_forecasts - observations)
        / (np.abs(observations) + np.abs(point_forecasts))
    )

    # Fix zero-observation values
    res[np.isnan(res)] = 0.

    return res


# ## MASE


def mase_score(observations, point_forecasts, horizon):
    mae_naive = mae_score(observations[:, horizon:], observations[:, 0:-horizon])
    mae_pred = mae_score(observations, point_forecasts)
    return mae_pred / mae_naive