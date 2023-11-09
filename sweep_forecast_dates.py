"""
A script to produce forecast for a set of dates.
"""

import copy
import shutil
import time
from collections import OrderedDict

# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from colorama import Fore, Style

import rtrend_tools.data_io as dio
import rtrend_tools.interpolate as interp
import rtrend_tools.scoring as score
import rtrend_tools.synthesis as synth
import rtrend_tools.rt_estimation as mcmcrt
import rtrend_tools.visualization as vis
from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES, CDC_ALPHAS
from rtrend_tools.forecast_structs import MosqData, ForecastExecutionData, ForecastOutput, \
    ForecastPost, TgData
from rtrend_tools.preprocessing import DENOISE_CALLABLE, NOISE_CLASS, regularize_negatives_from_denoise
from rtrend_tools.utils import map_parallel_or_sequential  # , make_dir_from
from toolbox.file_tools import read_config_file, str_to_bool_safe

# Directories to store temporary MCMC files
MCMC_IN_DIR = "mcmc_inputs/sweepdates_in/"
MCMC_OUT_DIR = "mcmc_outputs/sweepdates_out/"
MCMC_LOG_DIR = "mcmc_logs/sweepdates_log/"

WEEKLEN = interp.WEEKLEN
MAX_PIPE_STEPS = 100  # Max. number of overall steps in the forecast pipeline to declare failure
PIPE_STAGE = ("preproc", "interp", "r_estim", "r_synth", "c_reconst")  # Sequence of stages of the forecast


def main():

    args = parse_program_args()
    params = import_parse_param_file(args)
    mosq, tg = import_essential_data(params)

    dates_ens = prepare_dates_and_ensemble(params, mosq, tg)

    post_list = run_main_procedure(params, args, mosq, tg, dates_ens)
    aggregate_and_export(params, args, post_list, dates_ens)

    vis.make_plot_tables_sweepdates(post_list, mosq, tg, dates_ens, params, args)


def parse_program_args():
    parser = ArgumentParser()

    parser.add_argument("param_file", type=str)   # File with input parameters.
    parser.add_argument("output_dir", type=str)   # Directory for outputs.

    # Overrider flags
    parser.add_argument("-n", "--ncpus", type=int)
    parser.add_argument("--roi-size", type=int)  # Preprocessing ROI in weeks (must be greater than MCMC Roi size).
    parser.add_argument("--nweeks-fore", type=int)
    parser.add_argument("--preproc-method")
    parser.add_argument("--noise-type")
    parser.add_argument('--use-denoised', action=BooleanOptionalAction)  # --no-use-denoised
    parser.add_argument("-c", "--cutoff", type=float)  # Cutoff for lowpass filter method
    parser.add_argument("--noise-coef", type=float)    # Noise multiplicative coefficient
    parser.add_argument('--export', action=BooleanOptionalAction)  # --no-export
    parser.add_argument("--plots", action=BooleanOptionalAction, default=False)  # --no-plots
    parser.add_argument("--export-rtfore",  # Date for which the complete R(t) should be exported
                        type=pd.Timestamp, default=None)
    parser.add_argument("--export-noise-params", action=BooleanOptionalAction, default=None)  # Default none to avoid overriding

    args = parser.parse_args()

    return args


def import_parse_param_file(args):
    """
    Read the contents of the input parameter file. Then parse and distribute its
    parameters over use-specific dictionaries.
    Overrides parameters according to program call arguments.

    Returns
    -------
    ParametersBunch
    """

    input_dict = ipd = read_config_file(args.param_file)

    preproc_params = dict()
    interp_params = dict()
    mcmc_params = dict()
    synth_params = dict()
    recons_params = dict()
    misc_params = dict()

    # --- Preprocessing params
    d = preproc_params
    d["method"] = ipd["preproc_method"]
    d["noise_type"] = ipd["noise_type"]
    d["noise_table_path"] = ipd.get("noise_table_path", None)
    d["noise_coef"] = float(ipd.get("noise_coef", 1.0))
    d["noise_seed"] = int(ipd.get("noise_seed", 0))
    d["noise_mean"] = float(ipd["noise_mean"]) if "noise_mean" in ipd else None
    d["noise_std"] = float(ipd["noise_std"]) if "noise_std" in ipd else None
    d["roi_len"] = int(ipd["preproc_roi_len"])
    d["poly_degree"] = int(ipd.get("poly_degree", 3))
    d["cutoff"] = float(ipd.get("cutoff", 0.20))
    d["rollav_window"] = int(ipd.get("rollav_window", 4))

    # --- Interpolation parameters
    d = interp_params
    d["rel_smooth_fac"] = float(ipd.get("interp_smooth_fac", 0.01))
    d["spline_degree"] = int(ipd.get("spline_degree", 3))
    d["use_denoised"] = str_to_bool_safe(ipd.get("use_denoised", "False"))

    # --- Past R(t) estimation (MCMC) parameters
    d = mcmc_params
    d["nsim"] = int(ipd.get("nsim", 20000))  # Number of simulations (including 10,000 burn-in period)
    d["seed"] = int(ipd.get("mcmc_seed", 20))
    d["sigma"] = float(ipd.get("mcmc_sigma", 0.05))
    d["roi_len_days"] = int(ipd["mcmc_roi_len_days"])
    d["use_tmp"] = str_to_bool_safe(ipd.get("mcmc_use_tmp", "True"))

    # --- Future R(t) synthesis parameters
    d = synth_params
    d["method"] = ipd["synth_method"]
    d["tg_method"] = ipd.get("tg_synth_method", "near_past_avg")
    d["seed"] = int(ipd.get("synth_seed", 10))

    d["q_low"] = float(ipd.get("q_low", 0.40))
    d["q_hig"] = float(ipd.get("q_hig", 0.60))
    d["ndays_past"] = int(ipd.get("ndays_past", 28))  # Number of days (backwards) to consider in synthesis.
    d["r_max"] = float(ipd.get("r_max", -1))  # Clamp up to these R values (for the average)
    d["k_start"] = float(ipd.get("k_start", 1.0))  # Static Ramp method: starting coefficient
    d["k_end"] = float(ipd.get("k_end", 1.0))  # Static Ramp method: starting coefficient
    d["r1_start"] = float(ipd.get("r1_start", 1.00))  # Dynamic ramp: target starting value for R = 1
    d["r2_start"] = float(ipd.get("r2_start", 2.00))  # Dynamic ramp: target starting value for R = 2
    d["r1_end"] = float(ipd.get("r1_end", 1.00))  # Dynamic ramp: target ending value for R = 1
    d["r2_end"] = float(ipd.get("r2_end", 2.00))  # Dynamic ramp: target ending value for R = 2
    d["rmean_top"] = float(ipd.get("rmean_top", 1.40))  # Rtop ramp: starting value.
    d["max_width"] = float(ipd.get("rtop_max_width", 0.15))  # Rtop ramp: quantile interval width under r_top.
    d["r_fname"] = ipd.get("r_fname", None)
    d["tg_file"] = ipd.get("tg_synth_fname", None)
    d["center"] = float(ipd.get("rn_center", 1.0))
    d["sigma"] = float(ipd.get("rn_sigma", 0.05))
    d["num_samples"] = int(ipd.get("rn_num_samples", 1000))

    d = recons_params
    d["seed"] = int(ipd.get("recons_seed", 30))

    # --- Other general parameters (program parameters)
    d = misc_params
    d["species"] = ipd["species"]
    d["dates_first"] = pd.Timestamp(ipd["use_dates_first"])
    d["dates_last"] = pd.Timestamp(ipd["use_dates_last"])
    d["step_weeks"] = int(ipd.get("step_weeks", 1))
    d["nweeks_fore"] = int(ipd["nweeks_fore"])
    d["truth_data_fname"] = ipd["truth_data_fname"]
    d["tg_data_fname"] = ipd["tg_data_fname"]
    d["apply_scores"] = str_to_bool_safe(ipd.get("apply_scores", "True"))
    d["do_plots"] = str_to_bool_safe(ipd.get("do_plots", False))  # Threaded plot, won't show anyway
    d["ncpus"] = int(ipd.get("ncpus", 1))
    d["tg_max"] = int(ipd["tg_max"])
    d["ct_weekly_max"] = float(ipd.get("ct_weekly_max", np.inf))
    d["export"] = str_to_bool_safe(ipd.get("export", "True"))  # Can suppress all outputs
    d["export_fore_quantiles"] = str_to_bool_safe(ipd.get("export_fore_quantiles", "True"))
    d["export_preprocess"] = str_to_bool_safe(ipd.get("export_preprocess", "True"))
    d["export_rt_pastfore"] = str_to_bool_safe(ipd.get("export_rt_pastfore", "True"))
    d["export_noise_params"] = str_to_bool_safe(ipd.get("export_noise_params", "False"))

    # --- PARAMETER CHECKS
    if preproc_params["roi_len"] * WEEKLEN < mcmc_params["roi_len_days"]:
        raise ValueError(f"Hey, preprocessing ROI ({preproc_params['roi_len']}w) "
                         f"must be longer than MCMC ROI ({mcmc_params['roi_len_days']}d).")

    # --- Override parameters through program call arguments
    if args.ncpus is not None:
        misc_params["ncpus"] = args.ncpus

    if args.roi_size is not None:  # Preprocessing ROI length in weeks
        preproc_params["roi_len"] = args.roi_size

    if args.nweeks_fore is not None:
        misc_params["nweeks_fore"] = args.nweeks_fore

    if args.preproc_method is not None:
        preproc_params["method"] = args.preproc_method

    if args.noise_type is not None:
        preproc_params["noise_type"] = args.noise_type

    if args.use_denoised is not None:
        interp_params["use_denoised"] = args.use_denoised

    if args.cutoff is not None:
        preproc_params["cutoff"] = args.cutoff

    if args.noise_coef is not None:
        preproc_params["noise_coef"] = args.noise_coef

    if args.export is not None:
        misc_params["export"] = args.export

    if args.plots is not None:
        misc_params["do_plots"] = args.plots

    if args.export_noise_params is not None:
        misc_params["export_noise_params"] = args.export_noise_params

    return ParametersBunch(preproc_params, interp_params, mcmc_params, synth_params,
                           recons_params, misc_params, input_dict)


class ParametersBunch:
    def __init__(self, preproc_params: dict, interp_params: dict, mcmc_params: dict, synth_params: dict,
                 recons_params: dict, misc_params: dict, input_dict: dict):
        self.preproc = preproc_params
        self.interp = interp_params
        self.mcmc = mcmc_params
        self.synth = synth_params
        self.recons = recons_params
        self.misc = misc_params
        self.input = input_dict


def _aux_import_noise_data_with_checks(preproc):
    """"""
    if preproc["noise_table_path"] is None:
        raise ValueError("Hey, for this 'noise_type', the parameter 'noise_table_path' is required.")

    if not os.path.exists(preproc["noise_table_path"]):
        raise FileNotFoundError(f"Hey, noise_table_path = '{preproc['noise_table_path']} does not exist.")

    preproc["noise_table"] = pd.read_csv(preproc["noise_table_path"], index_col=0)


def import_essential_data(params: ParametersBunch):
    # Warns about opt out from exporting
    if not params.misc["export"]:
        print(Style.BRIGHT + 10 * "#" + " EXPORT SWITCH IS OFF " + 12 * "#" + Style.RESET_ALL)

    print("Importing and preprocessing data...")
    mosq = dio.load_mosquito_test_data(params.misc["truth_data_fname"])
    tg = dio.load_tg_dynamic_data(params.misc["tg_data_fname"], tg_max=params.misc["tg_max"])
    dio.remove_duplicated_mosq_entries_inplace(mosq)

    # Performance trick: imports file ensemble once
    if params.synth["method"] in ["file_ens", "file_as_slope"]:
        params.synth["r_fname"] = np.loadtxt(params.synth["r_fname"])

    # Performance trick: import noise data once
    if "normal_table" in params.preproc["noise_type"]:
        _aux_import_noise_data_with_checks(params.preproc)

    return mosq, tg


# noinspection PyUnusedLocal
def prepare_dates_and_ensemble(params: ParametersBunch, mosq: MosqData, tg: TgData):
    """Overall preprocessing. Defines the ensemble of dates to be used."""

    # print(mosq.df)
    # print(params.misc["dates_first"])
    dates = mosq.df[params.misc["dates_first"]:params.misc["dates_last"]]

    if params.misc["step_weeks"] > 1:
        idx = np.arange(0, dates.shape[0], params.misc["step_weeks"])
        dates = dates.iloc[idx]

    dates = dates.index

    return dates


def run_main_procedure(params: ParametersBunch, args, mosq: MosqData, tg: TgData,
                       dates: pd.DatetimeIndex):
    """Run forecast for all dates (present day) of the ensemble."""
    num_dates = dates.shape[0]
    output_dir = args.output_dir
    export_rtfore = args.export_rtfore

    #
    #
    def forecast_task(inputs):
        i_date, day_pres = inputs

        print(f"--------------({i_date + 1} of {num_dates}) Running for {day_pres.date()}")

        # ---------------- Forecast command
        fc: ForecastOutput = \
            forecast_this_date(day_pres, params, mosq, tg, params.misc["nweeks_fore"],
                               params.preproc, params.interp, params.mcmc, params.synth, params.recons)

        # Feedback from the forecast
        if fc is None:
            print(f"   [{day_pres.date()}] Returned empty, skipped.")
            return None
        print(f"   [{day_pres.date()}] Forecast done with {fc.num_ct_samples} c(t) samples.")

        # Postprocess and return
        return postprocess_forecast(fc, i_date, mosq, params, output_dir, export_rtfore)

    #
    #
    xt0 = time.time()
    result_list = map_parallel_or_sequential(
        forecast_task, enumerate(dates), ncpus=params.misc["ncpus"])
    xtf = time.time()
    print(f"SWEEP FORECAST TIME: {xtf - xt0:0.5f}s")

    return result_list


def callback_preprocess(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData):
    # Briefing
    # ---------------------------
    if exd.notes == "NONE":
        exd.method = "STD"
        denoise = DENOISE_CALLABLE[exd.preproc_params["method"]]
        fc.noise_obj = NOISE_CLASS[exd.preproc_params["noise_type"]](**exd.preproc_params)

    else:
        sys.stderr.write(f"\nHey, abnormal condition found in preproc briefing. (exd.notes = {exd.notes})\n")
        exd.stage = "ERROR"
        return

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------

    # PREPROCESSING ROI and other important properties of the time series
    mosq.week_roi_start = mosq.day_pres - pd.Timedelta(exd.preproc_params["roi_len"] - 1, "w")
    mosq.week_roi_end = mosq.day_pres
    mosq.day_roi_start = mosq.week_roi_start - pd.Timedelta(1, "w")
    mosq.day_roi_end = mosq.week_roi_end - pd.Timedelta(1, "d")
    mosq.day_roi_len = (mosq.day_roi_end - mosq.day_roi_start).days + 1

    exd.species_series = mosq.df[exd.species][mosq.week_roi_start:mosq.week_roi_end]
    tg.df_roi = tg.df.loc[mosq.day_roi_start:mosq.day_roi_end]

    if mosq.day_roi_len < tg.max:
        sys.stderr.write(
            Fore.YELLOW +
            f"[WARNING] ROI of length {mosq.day_roi_len} is shorter than the Tg maximum {tg.max}.\n" + Style.RESET_ALL)

    # Extract some days time-related values and containers
    fc.day_0 = mosq.day_roi_start  # First day of the daily ROI
    fc.day_pres = mosq.day_pres  # First day of forecast, one past the last of the daily ROI.
    fc.t_weekly = (exd.species_series.index - fc.day_0).days.array  # Weekly number of days as integers.

    # Denoise sequence
    denoise(exd, fc, tg, mosq, **exd.preproc_params)
    regularize_negatives_from_denoise(exd.species_series, fc.denoised_weekly)

    # Noise analysis and fit
    fc.noise_obj.fit(exd.species_series, fc.denoised_weekly)

    # Debriefing
    # ---------------------------
    if fc.denoised_weekly.min() < 0:  # Negative data from denoising procedure
        exd.stage, exd.notes = "ERROR", "negative"
        sys.stderr.write(Fore.YELLOW + "Negative value produced on denoising stage. Try another method.\n"
                         + Style.RESET_ALL)

    else:
        exd.stage, exd.notes = "interp", "NONE"  # Approve with default method


# noinspection PyArgumentList
# noinspection PyUnusedLocal
def callback_interpolation(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData):
    """
    Interpolate the mosquito data from weekly to daily.
    Some further preprocessing, like smoothening, can be done too.
    """
    # Briefing
    # ---------------------------
    exd.method = "STD"
    exd.log_current_pipe_step()

    # Execution
    # ---------------------------
    # Select between regular or denoised data
    fc.data_weekly = fc.denoised_weekly if exd.interp_params["use_denoised"] else exd.species_series

    # Run the interpolation
    fc.t_daily, fc.ct_past, fc.daily_spline, fc.float_data_daily = \
        interp.weekly_to_daily_spline(fc.t_weekly, fc.data_weekly, return_complete=True, **exd.interp_params)

    # Get more parameters related to the PREPROCESSING ROI.
    fc.past_daily_tlabels = fc.day_0 + pd.TimedeltaIndex(fc.t_daily, "D")
    fc.ndays_roi = fc.t_daily.shape[0]
    fc.ndays_fore = WEEKLEN * exd.nweeks_fore

    # Debriefing
    # ---------------------------
    # Check data integrity
    if fc.ct_past.min() < 0:
        print(Fore.YELLOW + f"[{fc.day_pres.date()}] HEY, negative counts ({fc.ct_past.min()}) found from "
                            f"interpolation." + Style.RESET_ALL)
        exd.stage = "ERROR"

    else:
        exd.stage, exd.notes = "r_estim", "NONE"  # Approve with default method


# noinspection PyUnusedLocal
def callback_r_estimation(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData):
    """
    IMPORTANT:
    This version of MCMC R(t) estimation uses its own ROI, shorter than the preprocessing ROI.
    That's for better preprocessing without impacting code performance (MCMC takes time).

    As a consequence, the array in rtm is shorter than fc.past_daily_tlabels
    """

    # Briefing
    # ---------------------------
    exd.method = "STD"
    exd.log_current_pipe_step()

    # Execution
    # ---------------------------

    # --- Further crop of preprocess ROI to MCMC ROI
    n = exd.mcmc_params["roi_len_days"]
    ct_past_crop = fc.ct_past[-n:]
    tg_df_roi_crop = tg.df_roi.iloc[-n:]

    # --- Definition of the temporary files
    if exd.mcmc_params["use_tmp"]:
        rng = np.random.default_rng()
        tail = f"tmp_{fc.day_pres.date()}_i{rng.integers(100000):05d}" + os.path.sep
    else:
        tail = f"{fc.day_pres.date()}" + os.path.sep

    exd.mcmc_params["in_prefix"] = os.path.join(MCMC_IN_DIR, tail)
    exd.mcmc_params["out_prefix"] = os.path.join(MCMC_OUT_DIR, tail)
    exd.mcmc_params["log_prefix"] = os.path.join(MCMC_LOG_DIR, tail)

    # --- BYPASS: if file_ens, running is not needed. Use pre-imported array.
    if exd.synth_params["method"] == "file_ens":
        print("BYPASSING MCMC BY USING PRE-LOADED DATA")
        # Crop the region from the file ensemble
        rt_array = exd.synth_params["r_fname"]  # As done by the trick
        i_last = fc.day_pres.day_of_year - 1
        i_days = np.arange(i_last - exd.mcmc_params["roi_len_days"], i_last) % rt_array.shape[1]  # Cycles through the file.
        fc.rtm = mcmcrt.McmcRtEnsemble(rt_array[:, i_days])  # Do the crop and create RT object

    else:
        # --- MCMC CALL
        try:
            fc.rtm = mcmcrt.run_mcmc_rt(
                ct_past_crop, tg_df_roi_crop, species=exd.species, **exd.mcmc_params)

        except mcmcrt.McmcError:
            exd.stage = "ERROR"  # Send error signal to pipeline

        finally:
            if exd.mcmc_params["use_tmp"]:
                for name in ["in_prefix", "out_prefix", "log_prefix"]:
                    shutil.rmtree(exd.mcmc_params[name])

        if exd.stage == "ERROR":
            return

    # # --- Tmp files cleanup
    # if exd.mcmc_params["use_tmp"]:
    #     for name in ["in_prefix", "out_prefix", "log_prefix"]:  # Deletes temporary dirs
    #         shutil.rmtree(exd.mcmc_params[name])

    # Debriefing
    # ---------------------------
    exd.stage, exd.notes = "r_synth", "NONE"  # Approve with default method


# noinspection PyUnusedLocal
def callback_r_synthesis(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData):

    # Cumulative number inside ROI
    sum_roi = exd.species_series.sum()

    # General preprocessing
    if exd.synth_params["r_max"] < 0:
        exd.synth_params["r_max"] = None  # Deactivate r_max filtering.

    # Method decision tree
    if exd.notes == "NONE":  # DEFAULT CONDITION, does not fall into any of the previous
        # R
        fc.synth_name = exd.method = exd.synth_params["method"]
        synth_method = synth.SYNTH_METHOD[exd.method]

        # Tg
        fc.tg_synth_name = exd.synth_params["tg_method"]
        tg_synth_method = synth.TG_SYNTH_METHOD[fc.tg_synth_name]

    else:
        sys.stderr.write(f"\nHey, abnormal condition found in r_synthesis briefing. (exd.method = {exd.method})\n")
        exd.stage = "ERROR"
        return

    # Preprocessing for the file ensemble method
    if exd.method in ["file_ens", "file_as_slope"]:
        exd.synth_params["fname"] = exd.synth_params["r_fname"]
        exd.synth_params["i_pres"] = mosq.day_pres.day_of_year - 1  # Index of the day in year

    if fc.tg_synth_name in ["file_series"]:
        # exd.synth_params["tg_file"] = exd.synth_params["tg_fname"]
        exd.synth_params["day_pres"] = fc.day_pres

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------

    # Apply the method
    fc.rt_fore2d = synth_method(fc.rtm, fc.ct_past, fc.ndays_fore, **exd.synth_params)
    fc.tg_fore_shape, fc.tg_fore_rate = tg_synth_method(tg, fc.rtm, fc.ct_past, fc.ndays_fore, **exd.synth_params)

    # Debriefing
    # ----------
    if fc.rt_fore2d.shape[0] == 0:  # Problem: no R value satisfied the filter criterion
        # -()- Just skip
        print(Fore.YELLOW + f"\t[{fc.day_pres.date()}] empty synth R(t) sample." + Style.RESET_ALL)
        exd.stage = "ERROR"
        return

        # # -()- Use random normal.
        # exd.stage, exd.notes = "r_synth", "use_rnd_normal"
        # exd.synth_params["center"] = 1.1  # Arbitrarily chosen
        # return

    exd.stage, exd.notes = "c_reconst", "NONE"  # Approve with default method


# noinspection PyUnusedLocal
def callback_c_reconstruction(exd: ForecastExecutionData, fc: ForecastOutput, tg: TgData, mosq: MosqData):
    """"""
    # Briefing
    # ----------------------------
    exd.method = "STD"

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------

    # Create a time series of Tg gamma distributions
    tg.past_2d_array = synth.make_tg_gamma_matrix(tg.df_roi["Shape"].array, tg.df_roi["Rate"].array, tg.max)
    tg.fore_2d_array = synth.make_tg_gamma_matrix(fc.tg_fore_shape, fc.tg_fore_rate, tg.max)

    # Reconstruct
    fc.ct_fore2d = synth.reconstruct_ct_tgtable(fc.ct_past, fc.rt_fore2d, tg, **exd.recons_params)
    fc.ct_fore2d_weekly = interp.daily_to_weekly(fc.ct_fore2d)

    # Incorporate noise uncertainty
    fc.ct_fore2d_weekly = fc.noise_obj.generate(fc.ct_fore2d_weekly)

    # Debriefing
    # ---------------------------
    # Attribute number of samples
    fc.num_ct_samples = fc.ct_fore2d_weekly.shape[0]

    # If this point was reached, results are satisfactory
    exd.stage = "DONE"  # Approve the forecast


def forecast_this_date(date: pd.Timestamp, params: ParametersBunch, mosq: MosqData, tg: TgData, nweeks_fore,
                       preproc_params, interp_params, mcmc_params, synth_params, recons_params):
    """
    Performs the forecast pipeline for one date and species.
    Returns a ForecastOutput object, without postprocessing.
    """

    exd = ForecastExecutionData()
    fc = ForecastOutput()

    # Populate execution data (exd). Params are deep-copied to protect original master dictionary.
    exd.species = params.misc["species"]
    exd.nweeks_fore = nweeks_fore
    exd.preproc_params = copy.deepcopy(preproc_params)
    exd.interp_params = copy.deepcopy(interp_params)
    exd.mcmc_params = copy.deepcopy(mcmc_params)
    exd.synth_params = copy.deepcopy(synth_params)
    exd.recons_params = copy.deepcopy(recons_params)
    # exd.ct_cap = ct_cap

    mosq = copy.copy(mosq)  # TODO: test if data is not cross-modified in SEQ and PARALL execution
    tg = copy.copy(tg)
    mosq.day_pres = date

    # PIPELINE LOOP OF STAGES
    # ------------------------------------------------------------------------------------------------------------------

    # Initialize the
    exd.stage, exd.method, exd.notes = PIPE_STAGE[0], "STD", "NONE"

    for i_pipe_step in range(MAX_PIPE_STEPS):

        if exd.stage == "preproc":  # STAGE 0: Denoising and noise fitting.
            callback_preprocess(exd, fc, tg, mosq)

        if exd.stage == "interp":  # STAGE 1: C(t) weekly to daily interpolation
            callback_interpolation(exd, fc, tg, mosq)

        elif exd.stage == "r_estim":  # STAGE 2: R(t) MCMC estimation
            callback_r_estimation(exd, fc, tg, mosq)

        elif exd.stage == "r_synth":  # STAGE 3: Future R(t) synthesis
            callback_r_synthesis(exd, fc, tg, mosq)

        elif exd.stage == "c_reconst":  # STAGE 4: Future C(t) reconstruction
            callback_c_reconstruction(exd, fc, tg, mosq)

        elif exd.stage == "DONE":  # Successful completion of all stages
            break

        elif exd.stage == "ERROR":  # A problem occurred in previous pipe step.
            """HANDLE FAILURE HERE"""
            # Current policy for pipeline errors: skip species.
            sys.stderr.write(f"\n\t[{date} will be skipped.]\n")
            return None

        else:
            raise ValueError(f"[ERROR {mosq.day_pres.date()}] Hey, exd.stage = \"{exd.stage}\" unrecognized.")

    # USE THIS TO CHECK LOGS
    # print(f"({mosq.day_pres.date()}) exd.stage_log = {exd.stage_log}")  # Check log of completed stages

    return fc


# noinspection PyUnusedLocal
def postprocess_forecast(
        fc: ForecastOutput, i_d, mosq: MosqData, params: ParametersBunch,
        output_dir, export_rtfore
):
    """
    Prepare data for final aggregation and for plot reports.
    Returns a ForecastPost object, which is lightweight and can be returned from the forecast loop to the main scope.
    """
    post = ForecastPost()
    post.species = params.misc["species"]
    post.synth_name = fc.synth_name
    nweeks_fore = params.misc["nweeks_fore"]

    # --- FOR SPINOFF WORK: exports the R(t) of this date
    if fc.day_pres == export_rtfore:
        fname = os.path.join(output_dir, f"{fc.day_pres.date().isoformat()}_rt-matrix.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(fname, "w") as file:
            # Write file metadata:
            # file.write(f"{fc.rt_fore2d.shape[0]}\n")  # Number of samples (rows)
            # file.write(f"{fc.rt_fore2d.shape[1]}\n")  # Number of dates (columns)
            header = f"{fc.rt_fore2d.shape[0]}\n{fc.rt_fore2d.shape[1]}"  # nsamples, ndates

            # fc.rt_fore2d.tofile(file, sep=",", format="%f")
            np.savetxt(file, fc.rt_fore2d, delimiter=",", fmt="%f", header=header, comments="")

            print(Fore.YELLOW + f"=========== Exported R(t) fore to `{fname}`" + Style.RESET_ALL)

    # --- Forecast data statistics
    post.quantile_seq = CDC_QUANTILES_SEQ
    post.num_quantiles = NUM_QUANTILES

    # Filter trajectories that exceed the desired value, then calculate quantiles
    filtered = synth.filter_ct_trajectories_by_max(
        fc.ct_fore2d_weekly, [params.misc["ct_weekly_max"]],
        min_required=10  # ARBITRARY number of minimum required trajectories to apply the filter
    )
    post.weekly_quantiles = synth.calc_tseries_ensemble_quantiles(filtered)

    # if post_coefs is not None:  # Apply post-coefficient for each week, if informed
    #     post.weekly_quantiles *= post_coefs
    post.num_ct_samples = filtered.shape[0]

    # --- Preprocessing data
    post.denoised_weekly = fc.denoised_weekly

    # Interpolation
    post.data_weekly = fc.data_weekly
    post.daily_spline = fc.daily_spline
    post.t_daily = fc.t_daily
    post.ct_past = fc.ct_past
    post.float_data_daily = fc.float_data_daily

    # Noie object
    post.noise_obj = fc.noise_obj

    # Stats of the past R(t) from MCMC
    post.rt_past_mean = fc.rtm.get_avg()
    post.rt_past_median = fc.rtm.get_median()
    post.rt_past_loquant, post.rt_past_hiquant = fc.rtm.get_quantiles()

    # Stats of the R(t) synthesis
    post.rt_fore_median = np.median(fc.rt_fore2d, axis=0)
    post.rt_fore_loquant = np.quantile(fc.rt_fore2d, 0.025, axis=0)
    post.rt_fore_hiquant = np.quantile(fc.rt_fore2d, 0.975, axis=0)

    # Datetime-like array of week day stamps
    post.day_0, post.day_pres = fc.day_0, fc.day_pres
    post.day_fore = fc.day_pres + pd.Timedelta(nweeks_fore, "W")
    post.past_daily_tlabels = fc.past_daily_tlabels
    post.fore_daily_tlabels = pd.date_range(fc.day_pres, periods=WEEKLEN * nweeks_fore, freq="D")
    post.fore_time_labels = \
        pd.DatetimeIndex((fc.day_pres + pd.Timedelta((i + 1) * WEEKLEN, "D") for i in range(nweeks_fore)))

    post.ground_truth_df = ground_truth_df = mosq.df[post.species].reindex(post.fore_time_labels)  # No key error, replace by NaN

    # Forecast scoring! Assumes that future dates are known
    if params.misc["apply_scores"]:
        # --------------------------------------
        # DISTRIBUTIONAL (quantile-based) SCORES
        # --------------------------------------

        # ground_truth_df = mosq.df[post.species].loc[post.fore_time_labels]  # Key error if not present!
        q_dict = {q: vals for (q, vals) in zip(post.quantile_seq, post.weekly_quantiles)}

        # Weighted Interval Scores (WIS)
        post.score_wis, sharp, post.score_wis_calib = score.weighted_interval_score_fast(
            ground_truth_df.values, CDC_ALPHAS, q_dict
        )  # Change this indexing to get SHARPNESS and CALIBRATION separately

        post.score_rwis = post.score_wis / ground_truth_df.values

        # Quantile coverages
        post.score_alpha = OrderedDict()
        post.score_alpha["50"] = score.alpha_covered_score(ground_truth_df.values, 0.50, q_dict)  # 50% interval
        post.score_alpha["90"] = score.alpha_covered_score(ground_truth_df.values, 0.10, q_dict)  # 90% interval
        post.score_alpha["95"] = score.alpha_covered_score(ground_truth_df.values, 0.05, q_dict)  # 95% interval

        # SQR: Simple Quantile Reach
        post.score_sqr = np.round(
            score.simple_quantile_reach_score_arraybased(
                ground_truth_df.values, post.quantile_seq, post.weekly_quantiles),
            decimals=6)

        # Log score (basically, log of the SQR)
        post.score_log = score.logscore_truncated(
            ground_truth_df.values, post.quantile_seq, post.weekly_quantiles)

        # -----------------------------------
        # POINT FORECAST (median) SCORES
        # -----------------------------------
        post.score_mae = score.ae_score(ground_truth_df.values, q_dict[0.50])
        post.score_mape = score.ape_score(ground_truth_df.values, q_dict[0.50])
        post.score_smape = score.sape_score(ground_truth_df.values, q_dict[0.50])

    return post


def aggregate_and_export(params: ParametersBunch, args, post_list: list[ForecastPost],
                         dates_ens):
    """"""
    print()
    print(6*"-------")
    print("POST OPERATIONS")

    # Preliminaries
    nweeks_fore = params.misc["nweeks_fore"]
    valid_mask = np.fromiter((post is not None for post in post_list), dtype=bool)
    post_varray = np.array(post_list)[valid_mask]  # Array of valid post objects
    #    ^ ^ Get forecasts with no error or skips

    # Export to file
    # --------------
    if not params.misc["export"]:
        return

    # --- Aggregation of forecast quantiles into a single DataFrame to export
    print("Aggregating quantiles to export")
    valid_dates = dates_ens[valid_mask]
    wks_ahead = 1 + np.arange(nweeks_fore)
    index = pd.MultiIndex.from_product([valid_dates, wks_ahead],
                                       names=["day_pres", "weeks_ahead"])
    content = np.concatenate([post.weekly_quantiles.T for post in post_varray], axis=0)
    quantiles_df = pd.DataFrame(content, index=index, columns=CDC_QUANTILES_SEQ)

    # --- Ground truth data
    quantiles_df["truth_vals"] = np.concatenate([post.ground_truth_df.values for post in post_varray])

    # --- Include scores
    if params.misc["apply_scores"]:
        # --- Distributional (quantile-based) scores
        # WIS total (calibration + sharpness)
        scores_wis = np.concatenate([post.score_wis for post in post_varray])
        quantiles_df.insert(0, "score_wis", scores_wis)

        # WIS calibration (accuracy only)
        wis_calib = np.concatenate([post.score_wis_calib for post in post_varray])
        quantiles_df.insert(1, "score_wis_calib", wis_calib)

        # Wis divided by observed values.
        scores_rwis = np.concatenate([post.score_rwis for post in post_varray])
        quantiles_df.insert(2, "score_rel_wis", scores_rwis)

        # Quantile coverages
        for i, alpha in enumerate(["50", "90", "95"]):
            scores_alpha = np.concatenate([post.score_alpha[alpha] for post in post_varray])
            quantiles_df.insert(i, f"score_alpha_{alpha}", scores_alpha)

        # SQR (outermost quantile)
        scores_sqr = np.concatenate([post.score_sqr for post in post_varray])
        quantiles_df.insert(3, "score_sqr", scores_sqr)

        # Log score
        scores_log = np.concatenate([post.score_log for post in post_varray])
        quantiles_df.insert(4, "score_log", scores_log)

        # --- Point forecast (median) scores
        # MAE, MAPE, sMAPE
        for i, name in enumerate(["mae", "mape", "smape"]):
            scores_ = np.concatenate([getattr(post, f"score_{name}") for post in post_varray])
            quantiles_df.insert(8 + i, f"score_{name}", scores_)

    # --- Include other info in the header
    params.input["input_file"] = args.param_file

    # --- Export
    q_fname = os.path.join(args.output_dir, "fore_quantiles.csv")
    p_fname = os.path.join(args.output_dir, "preprocess.csv")
    rpast_fname = os.path.join(args.output_dir, "rt_past.csv")
    rfore_fname = os.path.join(args.output_dir, "rt_fore.csv")
    noise_params_fname = os.path.join(args.output_dir, "noise_params.csv")

    if params.misc["export_fore_quantiles"]:
        dio.export_sweep_forecast(q_fname, quantiles_df, params.input)
        print(f"Exported quantiles to:\t{q_fname}")
    if params.misc["export_preprocess"]:
        dio.export_sweep_preprocess(p_fname, post_varray)
    if params.misc["export_rt_pastfore"]:
        # OBS: does not export if method is file ensemble, which uses precalculated
        # export_past = params.synth["method"] != "file_ens"  # No R(t) past to export if method is file_ens
        dio.export_sweep_rt(rpast_fname, rfore_fname, post_varray, params.mcmc["roi_len_days"])
    if params.misc["export_noise_params"]:
        dio.export_sweep_noise_params(noise_params_fname, post_varray)


if __name__ == "__main__":
    main()
