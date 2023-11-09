"""
Module with utilities for preprocessing the data (after ROI cut, before interpolation).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from rtrend_tools.forecast_structs import ForecastExecutionData, ForecastOutput, TgData, MosqData, AbstractNoise


# ----------------------------------------------------------------------------------------------------------------------
# AUX METHODS
# ----------------------------------------------------------------------------------------------------------------------
def apply_lowpass_filter(signal_array: np.ndarray, cutoff=0.4, order=2, ff_kwargs=None):
    if ff_kwargs is None:
        ff_kwargs = dict(method="gust")

    if len(signal_array.shape) > 1:
        warnings.warn(f"Hey, a {len(signal_array.shape):d}D array/df was passed to 'apply_lowpass_filter', but 1D is "
                      f"expected. The filtered data may be silently wrong.")
    # noinspection PyTupleAssignmentBalance
    butter_b, butter_a = butter(order, cutoff, btype='lowpass', analog=False)
    return filtfilt(butter_b, butter_a, signal_array, **ff_kwargs)


def apply_lowpass_filter_pdseries(signal_series: pd.Series, cutoff=0.4, order=2, ff_kwargs=None):
    """Same as apply_lowpass_filter, but accepts a pandas series and returns an equally indexed series."""
    return pd.Series(apply_lowpass_filter(signal_series.values, cutoff=cutoff, order=order, ff_kwargs=ff_kwargs),
                     index=signal_series.index)


def regularize_negatives_from_denoise(species_series: pd.Series, denoised_series: pd.Series, inplace=True):
    """
    As filtering and other denoising methods can produce negative values, this method eliminates it using some
    hardcoded criterion.
    """
    if inplace:
        out = denoised_series
    else:
        out = denoised_series.copy()

    neg_mask = denoised_series < 0.

    # # -()- Fallback negative denoised values to the original time series
    # out.loc[neg_mask] = species_series.loc[neg_mask]

    # -()- Replace by the central rolling average
    rollav_window = 8
    if neg_mask.any():
        rollav = species_series.rolling(rollav_window, center=True).mean()
        out.loc[neg_mask] = rollav.loc[neg_mask]
        out.fillna(species_series, inplace=True)  # Fill NANs (edges of the rolling sample) with original values

    return out


# ----------------------------------------------------------------------------------------------------------------------
# DENOISING METHODS (remember to update DENOISE CALLABLE numdict)
# ----------------------------------------------------------------------------------------------------------------------
# noinspection PyUnusedLocal
def denoise_lowpass(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData,
                    filt_order=2, cutoff=0.4, **kwargs):
    fc.denoised_weekly = pd.Series(apply_lowpass_filter(exd.species_series.values, cutoff, filt_order),
                                   index=exd.species_series.index)


# noinspection PyUnusedLocal
def denoise_polyfit(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData,
                    poly_degree=3, **kwargs):
    poly_coef, poly_resid = np.polyfit(fc.t_weekly, exd.species_series.values, deg=poly_degree, full=True)[0:2]
    poly_f = np.poly1d(poly_coef)  # Polynomial callable class.
    fc.denoised_weekly = pd.Series(poly_f(fc.t_weekly), index=exd.species_series.index)


# noinspection PyUnusedLocal
def denoise_rolling_average(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData,
                            rollav_window=4, **kwargs):
    fc.denoised_weekly = exd.species_series.rolling(rollav_window).mean()  # Rolling average
    fc.denoised_weekly[:rollav_window-1] = exd.species_series[:rollav_window-1]  # Fill NAN values with original ones
    # fc.denoised_weekly[:] *= exd.species_series[-1] / data_weekly[-1] if data_weekly[-1] else 1
    # #    Rescale to match last day


# Dummy method, only redirects to the original time series.
# noinspection PyUnusedLocal
def denoise_none(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData, **kwargs):
    fc.denoised_weekly = exd.species_series


# Dictionary of denoising methods.
DENOISE_CALLABLE = {
    "polyfit": denoise_polyfit,
    "lowpass": denoise_lowpass,
    "rollav": denoise_rolling_average,
    "none": denoise_none,
}


# ----------------------------------------------------------------------------------------------------------------------
# NOISE FIT AND SYNTH (GENERATE) METHODS
# ----------------------------------------------------------------------------------------------------------------------


class NormalMultNoise(AbstractNoise):
    """Normal (Gaussian) noise, equal for positive and negative deviations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = None
        self.std = None
        self.coef = kwargs.get("noise_coef", 1.0)  # Multiply noise by this coefficient
        self.seed = kwargs.get("noise_seed", None)  # Seed
        self._rng = np.random.default_rng(self.seed)

    def fit(self, data: pd.Series, denoised: pd.Series):
        reldev = calc_relative_dev_sample(data, denoised)

        self.mean = reldev.mean()
        self.mean = np.sign(self.mean) * min(abs(self.mean), 0.5)  # Clamp by +-0.5

        # self.std = reldev.std() / 2.  # -()- Use appropriate standard deviation
        self.std = reldev.std()  # Use doubled standard deviation
        self.std = np.sign(self.std) * min(abs(self.std), 1)  # Clamp by +-1

    def generate(self, new_denoised: np.ndarray):
        noise = np.maximum(self.coef * self._rng.normal(self.mean, self.std, size=new_denoised.shape), -1.)  # Clamped above -1
        return new_denoised * (1. + noise)


class NormalZeromeanMultNoise(AbstractNoise):
    """Normal (Gaussian) noise with mean fixed as zero. Only the standard deviation is estimated."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = None
        self.std = None
        self.coef = kwargs.get("noise_coef", 1.0)  # Multiply noise by this coefficient
        self.seed = kwargs.get("noise_seed", None)  # Seed
        self._rng = np.random.default_rng(self.seed)

    def fit(self, data: pd.Series, denoised: pd.Series):
        reldev = calc_relative_dev_sample(data, denoised)

        self.mean = 0

        # self.std = reldev.std() / 2.  # -()- Use appropriate standard deviation
        self.std = reldev.std()  # Use doubled standard deviation
        self.std = np.sign(self.std) * min(abs(self.std), 1)  # Clamp by +-1

    def generate(self, new_denoised: np.ndarray):
        noise = np.maximum(self.coef * self._rng.normal(self.mean, self.std, size=new_denoised.shape), -1.)  # Clamped above -1
        return new_denoised * (1. + noise)


class NormalFixedMultNoise(AbstractNoise):
    """Normal (Gaussian) noise, equal for positive and negative deviations.
    This class receives pre-fixed parameters instead of fitting from a ROI.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = kwargs["noise_mean"]
        self.std = kwargs["noise_std"]
        self.coef = kwargs.get("noise_coef", 1.0)   # Multiply noise by this coefficient
        self.seed = kwargs.get("noise_seed", None)  # Seed
        self._rng = np.random.default_rng(self.seed)

        # Check required parameters, must not be None
        for name in ["noise_mean", "noise_std"]:
            if kwargs[name] is None:
                raise ValueError(f"Hey, parameter `{name}` is required for NormalFixedMultNoise noise class.")

    def fit(self, data: pd.Series, denoised: pd.Series):
        """Nothing done to fit. Parameters are predefined."""

    def generate(self, new_denoised: np.ndarray):
        noise = np.maximum(self.coef * self._rng.normal(self.mean, self.std, size=new_denoised.shape), -1.)  # Clamped above -1
        return new_denoised * (1. + noise)


class NormalFixedZeromeanMultNoise(AbstractNoise):
    """Normal (Gaussian) noise, equal for positive and negative deviations.
    Sets the mean to zero.
    This class receives pre-fixed parameters instead of fitting from a ROI.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = 0
        self.std = kwargs["noise_std"]
        self.coef = kwargs.get("noise_coef", 1.0)   # Multiply noise by this coefficient
        self.seed = kwargs.get("noise_seed", None)  # Seed
        self._rng = np.random.default_rng(self.seed)

        # Check required parameters, must not be None
        for name in ["noise_std"]:
            if kwargs[name] is None:
                raise ValueError(f"Hey, parameter `{name}` is required for NormalFixedMultNoise noise class.")

    def fit(self, data: pd.Series, denoised: pd.Series):
        """Nothing done to fit. Parameters are predefined."""

    def generate(self, new_denoised: np.ndarray):
        noise = np.maximum(self.coef * self._rng.normal(self.mean, self.std, size=new_denoised.shape), -1.)  # Clamped above -1
        return new_denoised * (1. + noise)


class NormalTableMultNoise(NormalFixedMultNoise):
    """Normal (Gaussian) noise, equal for positive and negative deviations.
    This class receives parameters from a table, using an interpolation method."""
    def __init__(self, **kwargs):

        # Extract table of parameter values from kwargs
        param_table = kwargs["noise_table"]

        # Match the index names of the table with existing parameters in kwargs
        if isinstance(param_table.index, pd.MultiIndex):  # MULTIINDEX TREATMENT
            raise NotImplementedError()

        else:  # SIMPLE INDEX TREATMENT
            idx = kwargs[param_table.index.name]

            # Check index value: extrapolations are discouraged
            if idx < param_table.index.min() or idx > param_table.index.max():
                raise ValueError(f"Hey, value of `{param_table.index.name}` = {idx} is out of the range of "
                                 f"the provided noise table.")

        # Interpolate based on the selected index value
        kwargs["noise_mean"] = np.interp(idx, param_table.index, param_table['mean'])
        kwargs["noise_std"] = np.interp(idx, param_table.index, param_table['std'])

        super().__init__(**kwargs)


class NormalTableZeromeanMultNoise(NormalFixedZeromeanMultNoise):
    """Normal (Gaussian) noise, equal for positive and negative deviations.
    Sets the mean to 0 instead of using the pre-fitted value.
    This class receives parameters from a table, using an interpolation method."""
    def __init__(self, **kwargs):

        # Extract table of parameter values from kwargs
        param_table = kwargs["noise_table"]

        # Match the index names of the table with existing parameters in kwargs
        if isinstance(param_table.index, pd.MultiIndex):  # MULTIINDEX TREATMENT
            raise NotImplementedError()

        else:  # SIMPLE INDEX TREATMENT
            idx = kwargs[param_table.index.name]

            # Check index value: extrapolations are discouraged
            if idx < param_table.index.min() or idx > param_table.index.max():
                raise ValueError(f"Hey, value of `{param_table.index.name}` = {idx} is out of the range of "
                                 f"the provided noise table.")

        # Interpolate based on the selected index value
        # kwargs["noise_mean"] = np.interp(idx, param_table.index, param_table['mean'])
        kwargs["noise_std"] = np.interp(idx, param_table.index, param_table['std'])

        super().__init__(**kwargs)


class NoneNoise(AbstractNoise):
    """Produces zero noise."""
    def generate(self, new_denoised):
        return new_denoised


NOISE_CLASS = {
    "normal": NormalMultNoise,
    "normal_fixed": NormalFixedMultNoise,
    "normal_fixed_zeromean": NormalFixedZeromeanMultNoise,
    "normal_table": NormalTableMultNoise,
    "normal_table_zeromean": NormalTableZeromeanMultNoise,
    "none": NoneNoise,
}


# Common methods for multiple classes
# ------------------------------------
def calc_relative_dev_sample(data: pd.Series, denoised: pd.Series):
    """
    Returns
    -------
    pd.Series
    """
    return (data - denoised) / denoised
