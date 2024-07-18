"""
Runs a naive baseline mosquito abundance forecast for a sequence of dates.

Meant to run with similar input files as those for `sweep_forecast_dates.py`.
"""
import copy
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from collections import OrderedDict

from colorama import Style
import numpy as np
import pandas as pd
import scipy
import scipy.stats

import rtrend_tools.data_io as dio
import rtrend_tools.interpolate as interp
import rtrend_tools.scoring as score
import rtrend_tools.synthesis as synth
import rtrend_tools.visualization as vis
from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES, CDC_ALPHAS
from rtrend_tools.forecast_structs import MosqData, BaselForecastOutput, BaselForecastExecutionData, BaselForecastPost
from rtrend_tools.utils import map_parallel_or_sequential
from toolbox.file_tools import read_config_file, str_to_bool_safe

WEEKLEN = interp.WEEKLEN


def main():

    args = parse_program_args()
    params = import_parse_param_file(args)
    mosq = import_essential_data(params)

    dates_ens = prepare_dates_and_ensemble(params, mosq)

    post_list = run_main_procedure(params, args, mosq, dates_ens)
    aggregate_and_export(params, args, post_list, dates_ens)

    # vis.make_plot_tables_sweepdates(post_list, mosq, tg, dates_ens, params, args)


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


def parse_program_args():
    parser = ArgumentParser()

    parser.add_argument("param_file", type=str)  # File with input parameters.
    parser.add_argument("output_dir", type=str)  # Directory for outputs.

    # Overrider flags
    parser.add_argument("-n", "--ncpus", type=int)
    parser.add_argument("--roi-size", type=int)  # Preprocessing ROI in weeks (must be greater than MCMC Roi size).
    parser.add_argument("--nweeks-fore", type=int)
    # parser.add_argument("--preproc-method")
    # parser.add_argument("--noise-type")
    # parser.add_argument('--use-denoised', action=BooleanOptionalAction)  # --no-use-denoised
    # parser.add_argument("-c", "--cutoff", type=float)  # Cutoff for lowpass filter method
    # parser.add_argument("--noise-coef", type=float)  # Noise multiplicative coefficient
    parser.add_argument('--export', action=BooleanOptionalAction)  # --no-export
    parser.add_argument("--plots", action=BooleanOptionalAction, default=False)  # --no-plots

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
    d["roi_len"] = int(ipd["preproc_roi_len"])
    #  ^ ^ Whether to force the data to be integer.
    d["use_integer_data"] = str_to_bool_safe(ipd.get("use_integer_data", False))

    # --- Interpolation parameters
    d = interp_params
    # d["rel_smooth_fac"] = float(ipd.get("interp_smooth_fac", 0.01))
    # d["spline_degree"] = int(ipd.get("spline_degree", 3))
    # d["use_denoised"] = str_to_bool_safe(ipd.get("use_denoised", "False"))

    # --- Past R(t) estimation (MCMC) parameters
    d = mcmc_params
    d["roi_len_days"] = int(ipd["mcmc_roi_len_days"])

    # --- Future R(t) synthesis parameters
    d = synth_params
    # d["method"] = ipd["synth_method"]
    # d["tg_method"] = ipd.get("tg_synth_method", "near_past_avg")
    d["seed"] = int(ipd.get("synth_seed", 10))
    d["num_samples"] = int(ipd.get("rn_num_samples", 1000))
    d["smooth_diff_dist"] = str_to_bool_safe(ipd.get("smooth_diff_dist", False))
    d["fit_normal"] = str_to_bool_safe(ipd.get("fit_normal", True))  # NAIVE: should it fit dist. to normal??
    #  ^ ^ Whether the distribution of differences should be smoothed before sampling.

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

    # if args.preproc_method is not None:
    #     preproc_params["method"] = args.preproc_method

    # if args.noise_type is not None:
    #     preproc_params["noise_type"] = args.noise_type
    #
    # if args.use_denoised is not None:
    #     interp_params["use_denoised"] = args.use_denoised
    #
    # if args.cutoff is not None:
    #     preproc_params["cutoff"] = args.cutoff
    #
    # if args.noise_coef is not None:
    #     preproc_params["noise_coef"] = args.noise_coef

    if args.export is not None:
        misc_params["export"] = args.export

    if args.plots is not None:
        misc_params["do_plots"] = args.plots

    return ParametersBunch(preproc_params, interp_params, mcmc_params, synth_params,
                           recons_params, misc_params, input_dict)


def import_essential_data(params: ParametersBunch):
    # Warns about opt out from exporting
    if not params.misc["export"]:
        print(Style.BRIGHT + 10 * "#" + " EXPORT SWITCH IS OFF " + 12 * "#" + Style.RESET_ALL)

    print("Importing and preprocessing data...")
    mosq = dio.load_mosquito_test_data(params.misc["truth_data_fname"])
    # tg = dio.load_tg_dynamic_data(params.misc["tg_data_fname"], tg_max=params.misc["tg_max"])
    dio.remove_duplicated_mosq_entries_inplace(mosq)

    # # Performance trick: imports file ensemble once
    # if params.synth["method"] in ["file_ens", "file_as_slope"]:
    #     params.synth["r_fname"] = np.loadtxt(params.synth["r_fname"])

    # # Performance trick: import noise data once
    # if "normal_table" in params.preproc["noise_type"]:
    #     _aux_import_noise_data_with_checks(params.preproc)

    return mosq


# noinspection PyUnusedLocal
def prepare_dates_and_ensemble(params: ParametersBunch, mosq: MosqData):
    """Overall preprocessing. Defines the ensemble of dates to be used."""

    dates = mosq.df[params.misc["dates_first"]:params.misc["dates_last"]]

    if params.misc["step_weeks"] > 1:
        idx = np.arange(0, dates.shape[0], params.misc["step_weeks"])
        dates = dates.iloc[idx]

    dates = dates.index

    return dates


def forecast_this_date(
        day_pres: pd.Timestamp, params: ParametersBunch, mosq: MosqData, diff_sr: pd.Series,
        nweeks_fore, preproc_params, interp_params, mcmc_params, synth_params, recons_params
):
    """
    Performs the forecast pipeline for one date and species.
    Returns a ForecastOutput object, without postprocessing.
    """

    exd = BaselForecastExecutionData()
    fc = BaselForecastOutput()

    mosq = copy.copy(mosq)
    mosq.day_pres = day_pres

    # Preprocessing – Find the predictive distribution from past data
    # ===============================================================
    exd.past_diff_sr = diff_sr.loc[:day_pres]

    # Smoothing (optional)
    # --------------------
    params.synth["smooth_diff_dist"] = False  # WATCH BREAK CHANGEPOINT TODO REMOVE!!!!

    if params.synth["smooth_diff_dist"]:

        # Calculate the difference distribution
        past_diff_hist = exd.past_diff_sr.value_counts().sort_index()
        cdist = past_diff_hist.cumsum()
        cdist /= cdist.iloc[-1]

        # Interpolate the cumulative
        cinterp = cdist.reindex(pd.RangeIndex(start=0, stop=cdist.index.max() + 1))
        cinterp[0] = 0
        cinterp.interpolate(method="linear", inplace=True)

        # Reconstruct the distribution, now smoothed by the cumulative interpolation
        exd.pred_dist = cinterp.diff()
        exd.pred_dist[0] = 0.

    elif params.synth["fit_normal"]:
        # --- Fit the differences ensamble to a normal distribution, which is used to sample from.
        max_z = 3.0  # Maximum z-score from the normal that is considered.
        stdev = exd.past_diff_sr.std()
        diff_values = np.arange(start=0, stop=max_z * stdev, step=1.0)
        exd.pred_dist = pd.Series(
            scipy.stats.norm.pdf(diff_values, loc=0.0, scale=stdev),
            index=diff_values,
        )
        exd.pred_dist /= exd.pred_dist.sum()

    else:  # Here we simply use the sample values as "distribution"
        exd.pred_dist = pd.Series(1, index=exd.past_diff_sr.values)
        exd.pred_dist /= exd.pred_dist.shape[0]

    # [[[Deal with negatives, if method is not linear]]]

    # No Smoothing
    # --------------------

    # Forecasting – Sample trajectories from the predictive distribution
    # ==================================================================
    rng = np.random.default_rng(seed=synth_params["seed"])
    fore_weekly_index = pd.date_range(start=day_pres + pd.Timedelta("1w"), freq="1w", periods=4)
    nsamples = synth_params["num_samples"]
    start_value = mosq.df[params.misc["species"]].loc[day_pres]

    # -----
    # Sample the differences and construct trajectories

    # Random differences (absolute)
    diff_ensemble = rng.choice(
        exd.pred_dist.index, replace=True,
        size=(nsamples, nweeks_fore),
        p=exd.pred_dist.values
    )

    # Random signals
    signal_mask = 2 * rng.integers(2, size=(nsamples, nweeks_fore)) - 1

    # Build forecast
    fore_array = (diff_ensemble * signal_mask).cumsum(axis=1) + start_value
    fore_array[fore_array < 0] = 0  # Truncate the simple way

    # Construct forecast df including the last observed data point
    dtype = int if params.preproc["use_integer_data"] else float
    fc.fore_df = pd.concat([
        pd.DataFrame(start_value, index=pd.RangeIndex(nsamples), columns=[day_pres], dtype=dtype),
        pd.DataFrame(fore_array, columns=fore_weekly_index, dtype=dtype)
    ], axis=1)

    return exd, fc


def postprocess_forecast(
        fc: BaselForecastOutput, exd: BaselForecastExecutionData,
        mosq: MosqData, params: ParametersBunch, output_dir
):
    """"""
    post = BaselForecastPost()

    # --- Forecast data statistics
    post.quantile_seq = CDC_QUANTILES_SEQ
    post.num_quantiles = NUM_QUANTILES

    post.fore_time_labels = fc.fore_df.columns[1:]
    post.weekly_quantiles = fc.fore_df.quantile(post.quantile_seq, axis=0)

    post.ground_truth_df = mosq.df[params.misc["species"]].reindex(post.fore_time_labels)

    if params.misc["apply_scores"]:
        # --------------------------------------
        # DISTRIBUTIONAL (quantile-based) SCORES
        # --------------------------------------
        fore_weekly_quantiles = post.weekly_quantiles[post.fore_time_labels]  # Select only future ones
        q_dict = fore_weekly_quantiles.T  # DataFrame approach

        # Weighted Interval Scores (WIS)
        post.score_wis, sharp, post.score_wis_calib = score.weighted_interval_score_fast(
            post.ground_truth_df.values, CDC_ALPHAS, q_dict
        )  # Change this indexing to get SHARPNESS and CALIBRATION separately

        post.score_rwis = post.score_wis / post.ground_truth_df.values

        # Quantile coverages
        post.score_alpha = OrderedDict()
        post.score_alpha["50"] = score.alpha_covered_score(post.ground_truth_df.values, 0.50, q_dict)  # 50% interval
        post.score_alpha["90"] = score.alpha_covered_score(post.ground_truth_df.values, 0.10, q_dict)  # 90% interval
        post.score_alpha["95"] = score.alpha_covered_score(post.ground_truth_df.values, 0.05, q_dict)  # 95% interval

        # SQR: Simple Quantile Reach
        post.score_sqr = np.round(
            score.simple_quantile_reach_score_arraybased(
                    post.ground_truth_df.values, post.quantile_seq, fore_weekly_quantiles),
            decimals=6)

        # Log score (basically, log of the SQR)
        post.score_log = score.logscore_truncated(
            post.ground_truth_df.values, post.quantile_seq, fore_weekly_quantiles)

        # -----------------------------------
        # POINT FORECAST (median) SCORES
        # -----------------------------------
        post.score_mae = score.ae_score(post.ground_truth_df.values, q_dict[0.50])
        post.score_mape = score.ape_score(post.ground_truth_df.values, q_dict[0.50])
        post.score_smape = score.sape_score(post.ground_truth_df.values, q_dict[0.50])

    return post


def run_main_procedure(params: ParametersBunch, args, mosq: MosqData,
                       dates: pd.DatetimeIndex):
    """Run forecast for all dates (present day) of the ensemble."""
    num_dates = dates.shape[0]
    output_dir = args.output_dir

    # Preprocessing – calculate weekly differences
    # ---
    diff_sr = mosq.df[params.misc["species"]].diff().abs().iloc[1:]
    if params.preproc["use_integer_data"]:
        diff_sr = diff_sr.round().astype(int)

    def forecast_task(inputs):
        i_date, day_pres = inputs

        print(f"--------------({i_date + 1} of {num_dates}) Running for {day_pres.date()}")

        # ---------------- Forecast command
        exd, fc = forecast_this_date(
            day_pres, params, mosq, diff_sr, params.misc["nweeks_fore"],
            params.preproc, params.interp, params.mcmc, params.synth, params.recons
        )

        # - - - - - -
        #
        # Postprocess and return
        return postprocess_forecast(fc, exd, mosq, params, output_dir)

    xt0 = time.time()
    result_list = map_parallel_or_sequential(
        forecast_task, enumerate(dates), ncpus=params.misc["ncpus"])
    xtf = time.time()
    print(f"SWEEP FORECAST TIME: {xtf - xt0:0.5f}s")

    return result_list


def aggregate_and_export(
        params: ParametersBunch, args,
        post_list: list[BaselForecastPost], dates_ens
):
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
        print("### EXPORT SKIPPED ###")
        return

    # print("Aggregating quantiles to export")
    valid_dates = dates_ens[valid_mask]
    wks_ahead = 1 + np.arange(nweeks_fore)
    index = pd.MultiIndex.from_product([valid_dates, wks_ahead],
                                       names=["day_pres", "weeks_ahead"])
    content = np.concatenate(
        [post.weekly_quantiles[post.fore_time_labels].T.values for post in post_varray], axis=0)

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

    if params.misc["export_fore_quantiles"]:
        dio.export_sweep_forecast(q_fname, quantiles_df, params.input)
        print(f"Exported quantiles to:\t{q_fname}")

    print(quantiles_df)


if __name__ == '__main__':
    main()
