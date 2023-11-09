"""
Struct-like classes used in the forecast pipeline and surroundings.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


class MosqData:
    """Data bunch for the mosquito population time series."""

    def __init__(self):
        self.df = None
        self.roi_df_weekly = None  # Weekly ROI view from df

        self.species_names = None
        self.num_species = None

        self.data_time_labels = None
        self.day_roi_start: pd.Timestamp = None  # Start of the daily ROI
        self.day_roi_end: pd.Timestamp = None  # Last day (included) of the daily ROI
        self.day_pres: pd.Timestamp = None  # "Present" day, first one to be forecasted.
        self.week_roi_start: pd.Timestamp = None  # Start of the weekly ROI. End of the first week.
        self.week_roi_end: pd.Timestamp = None  # Last day (included) of the weekly ROI. Same as self.day_pres.
        self.day_roi_len: int = None


class TgData:
    """Data bunch for the dynamic generation time."""

    def __init__(self):
        self.df = None
        self.df_roi = None  # Daily ROI view from df

        self.max = None

        self.past_2d_array = None  # Generation time for the ROI | a[i_t, i_s]
        self.fore_2d_array = None  #


class CDCDataBunch:

    def __init__(self):
        self.df: pd.DataFrame = None  # DataFrame with CDC data as read from the file.

        self.num_locs = None  # Number of unique locations
        self.loc_names = None  # List of unique location names
        self.loc_ids = None  # List of location ids (as strings)
        self.to_loc_id = None  # Conversion dictionary from location name to location id
        self.to_loc_name = None  # Conversion dictionary from location id to location name

        self.data_time_labels = None  # Array of dates present in the whole dataset.

        # Preprocessing data (not set during loading)
        self.day_roi_start: pd.Timestamp = None
        self.day_pres: pd.Timestamp = None

    def xs_state(self, state_name):
        return self.df.xs(self.to_loc_id[state_name], level="location")


class ForecastExecutionData:
    """
    Data for the handling of the forecast pipeline for a single state.
    Handled inside the function forecast_state().
    """

    def __init__(self):
        # Input parameters and data
        self.species = None
        self.species_series: pd.Series = None
        self.nweeks_fore = None
        self.except_params: dict = None
        # self.tg: TgData = None
        self.preproc_params: dict = None
        self.interp_params: dict = None
        self.smooth_params: dict = None
        self.mcmc_params: dict = None
        self.synth_params: dict = None
        self.recons_params: dict = None
        # self.ct_cap = None

        # Pipeline handling
        self.stage = None   # Determines the forecast stage to be run in the next pipeline step. Written at debriefing.
        self.method = None  # Determines the method to use in current stage. Written at briefing.
        self.notes = None   # Debriefing notes to the briefing of the next stage. Written at debriefing.
        self.stage_log = list()

    def log_current_pipe_step(self):
        """Writes current stage and method into the stage_log"""
        self.stage_log.append((self.stage, self.method))


class ForecastOutput:
    """Contains data from forecasting process in one state: from interpolation to weekly reaggregation."""

    def __init__(self):
        self.day_0: pd.Timestamp = None  # First day to have report (7 days before the first stamp in ROI)
        self.day_pres: pd.Timestamp = None  # Day of the current report.
        self.ndays_roi = None
        self.ndays_fore = None

        # Preprocessing results
        self.denoised_weekly: pd.Series = None  # Denoised weekly data, used for noise fit
        self.noise_obj: AbstractNoise = None  # Noise fit/synth object (AbstractNoiseObj)

        # Interpolation results
        self.data_weekly: pd.Series = None  # Weekly data (denoised or not) used into the forecast.
        self.t_weekly: np.ndarray = None  # Weekly array of days as integers.
        self.t_daily: np.ndarray = None
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        # self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.ct_past: np.ndarray = None
        self.float_data_daily: np.ndarray = None  # Float data from the spline of the cumulative sum
        self.daily_spline: UnivariateSpline = None

        # MCMC
        import rtrend_tools.rt_estimation as mcmcrt
        self.rtm: mcmcrt.McmcRtEnsemble = None

        # Synthesis
        self.synth_name = None
        self.tg_synth_name = None
        self.rt_fore2d: np.ndarray = None  # a[i_sample, i_t]
        self.num_ct_samples = None

        self.tg_fore_shape: np.ndarray = None  # (Daily array) Synthesized shape parameter of generation time
        self.tg_fore_rate: np.ndarray = None   # (Daily array) Synthesized rate parameter

        # Reconstruction
        self.ct_fore2d: np.ndarray = None
        self.ct_fore2d_weekly: np.ndarray = None  # a[i_sample, i_t]
        # self.weekly_quantiles: np.ndarray = None


class ForecastPost:
    """
    Contains post-processed data from the forecasting process in one state: the quantiles and other lightweight
    arrays that are sent back to the main scope.
    """
    def __init__(self):
        self.species = None
        self.synth_name = None

        self.ground_truth_df = None

        # Forecasted data
        self.quantile_seq = None  # Just CDC default sequence of quantiles
        self.num_quantiles = None
        self.daily_quantiles: np.ndarray = None   # Forecasted quantiles | a[i_q, i_day]
        self.weekly_quantiles: np.ndarray = None  # Forecasted quantiles | a[i_q, i_week]
        # self.samples_to_us: np.ndarray = None  # ct_fore_weekly samples to sum up the US series. | a[i_sample, i_week]

        # Preprocessing
        self.denoised_weekly: pd.Series = None

        # Interpolation
        self.data_weekly: pd.Series = None
        self.t_daily: np.ndarray = None
        self.ct_past: np.ndarray = None
        self.float_data_daily: np.ndarray = None
        self.daily_spline: UnivariateSpline = None

        # Noise object
        self.noise_obj = None

        # MCMC past R(t) stats
        self.rt_past_mean: np.ndarray = None
        self.rt_past_median: np.ndarray = None
        self.rt_past_loquant: np.ndarray = None
        self.rt_past_hiquant: np.ndarray = None

        # Synthesis forecast R(t) stats
        self.rt_fore_median: np.ndarray = None
        self.rt_fore_loquant: np.ndarray = None
        self.rt_fore_hiquant: np.ndarray = None

        # Time labels
        self.day_0: pd.Timestamp = None     # First day of ROI.
        self.day_pres: pd.Timestamp = None  # Day time stamp of present
        self.day_fore: pd.Timestamp = None  # Last day forecasted
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.fore_time_labels: pd.DatetimeIndex = None  # Weekly time stamps for forecast region

        # Scores (if calculated)
        # Quantile-based
        self.score_wis: np.ndarray = None  # WIS score for known forecasts
        self.score_rwis: np.ndarray = None  # Relative WIS score for known forecasts
        self.score_wis_calib: np.ndarray = None
        self.score_alpha: OrderedDict[np.ndarray] = None
        self.score_sqr: np.ndarray = None  # Simple Quantile Reach: the outermost IQR that misses the observation
        # Point
        self.score_mae: np.ndarray = None
        self.score_mape: np.ndarray = None
        self.score_smape: np.ndarray = None
        self.score_log: np.ndarray = None

        # Misc
        self.xt = None  # Just some counter for execution time


class SweepDatesOutput:
    """Combines data that is exported from sweep_forecast_dates.py."""
    def __init__(self):
        # Forecast data containers
        self.q_df: pd.DataFrame = None  # Main data frame with forecast quantiles for multiple present dates.
        self.score_df: pd.DataFrame = None  # Dataframe with scores, if created.
        self.prep_df: pd.DataFrame = None  # Preprocessing dataframe
        self.rpast_df: pd.DataFrame = None  # Past R(t) data
        self.rfore_df: pd.DataFrame = None  # Forecast (synth) R(t) data
        self.dates_ens: pd.DatetimeIndex = None  # Index of dates used as day_pres

        self.input_dict: dict = None  # Raw input dict (from header), all values are strings.
        self.mosq: MosqData = None  # Truth data for mosquito abundance

        # Parameters interpreted from the input dictionary
        self.apply_scores: bool = None
        self.species: str = None
        self.nweeks_fore: int = None

        # Preprocessed stuff: only set if mosq data is present.
        self.species_series: pd.Series = None  # Abundance time series with the selected mosquito species
        self.truth_vals: pd.Series = None      # Species series but with an index that's aligned with self.score_df


# class USForecastPost:
#     """Special post process object for the US time series."""
#
#     def __init__(self):
#         self.weekly_quantiles: np.ndarray = None  # Forecasted quantiles | a[i_q, i_week]
#         self.daily_quantiles: np.ndarray = None   # Forecasted quantiles | a[i_q, i_day]
#         self.num_quantiles = None
#
#         self.day_0: pd.Timestamp = None     # First day of ROI.
#         self.day_pres: pd.Timestamp = None  # Day time stamp of present
#         self.day_fore: pd.Timestamp = None  # Last day forecasted
#         self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily dates
#         self.fore_time_labels: pd.DatetimeIndex = None  # Weekly dates for the forecasted series.


# Polymorphic class for noise fit'n synth
class AbstractNoise:

    def __init__(self, **kwargs):
        pass

    def fit(self, data: pd.Series, denoised: pd.Series):
        """Abstract method. Extracts parameters from the (noisy) data and denoised series. Feeds inner variables."""

    def generate(self, new_denoised: np.ndarray):
        """Abstract method. Must return a new data with calibrated noise incorporated."""
        raise NotImplementedError
