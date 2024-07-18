"""
This script tries to find the parameters that produce the best forecast for a given list of
input files.
"""
import copy
import datetime
import itertools
import os
import subprocess
import time
from argparse import ArgumentParser, ArgumentTypeError
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style
from scipy.stats.qmc import LatinHypercube

from rtrend_tools.data_io import load_sweep_forecast, aux_make_truth_vals_aligned
from rtrend_tools.forecast_structs import SweepDatesOutput
from rtrend_tools.utils import make_dir_from, map_parallel_or_sequential
from toolbox.file_tools import (
    read_config_file, write_config_file, seconds_to_hhmmss,
    prepare_dict_for_yaml_export
)


FORECAST_CALL = "python sweep_forecast_dates.py"  # Callable to forecast for one location.

TMP_INPUT_FMT = "input_params/tmp/optim_{:d}_{:06d}.in"
OUT_SUBDIR_FMT = "opt_{}"  # Prefix for each directory that stores the forecasts of each step

N_INPUT_CPUS = 3  # How many workers to have running each input file.
N_SWEEP_CPUS = 1  # How many processes within the forecast call (swee


class ProgramParams:

    # # # -()- Parameters to change. BROAD. VEEEEEERY BROAD FOR ALL PARAMETERS. Coarse search. Keep it the same btwn locations.
    # vary_params = OrderedDict(
    #     cutoff=[0.05, 0.60],
    #     # rt_width=[0.10, 1.0],
    #     mcmc_scaling_factor=[1., 500.],
    #     # noise_coef=[0.50, 3.0],
    #     # preproc_roi_len=list(range(7, 35)),
    #     # ndays_past=[5, 21],
    # )
    # optim_method = "lhs"

    # # -()- Parameters to change. FINE, Can be modified.
    # vary_params = OrderedDict(
    #     cutoff=[0.17, 0.25],
    #     mcmc_scaling_factor=[150., 450],
    # )
    # stepsize_params["cutoff"] = 0.005
    # stepsize_params["mcmc_scaling_factor"] = 1.0
    # optim_method = "gridsearch"  # lhs, gridsearch

    # # -()- TWO STEP (1): Cutoff calibration with WIS
    # vary_params = OrderedDict(
    #     cutoff=np.arange(0.05, 0.60, 0.005),
    # )
    # optim_method = "gridsearch"
    # score_weights = defaultdict(lambda: 0.0)  # Use default dict if you're confident with the names
    # score_weights["accuracy"] = score_weights["sharpness"] = 1.0

    # -()- TWO STEP (2): Scaling calibration with coverage
    vary_params = OrderedDict(
        mcmc_scaling_factor=np.arange(1., 100., 1),
    )
    optim_method = "gridsearch"
    score_weights = defaultdict(lambda: 0.0)  # Use default dict if you're confident with the names
    score_weights["coverage_50"] = score_weights["coverage_95"] = 1.0

    #
    # -----

    # # --- Los Angeles – No FAS (static ramp)
    # vary_params = OrderedDict(
    #     # rt_width=[0.25, 0.75],
    #     # cutoff=[0.20, 0.35],  # ___ 1st round
    #     rt_width=[0.25, 0.45],
    #     cutoff=[0.19, 0.23],  # ___ 2nd round
    # )

    # # --- Rnd Normal synth
    # vary_params = OrderedDict(
    #     # rn_sigma=[0.01, 0.40],
    #     # cutoff=[0.05, 0.70],  # ___ 1st round
    #     rn_sigma=[0.10, 0.33],
    #     cutoff=[0.45, 0.68],  # ___ 2nd round
    # )

    # # --- Key West 2nd round
    # vary_params = OrderedDict(
    #     rt_width=[0.3254808585025505, 0.5818624250322726],
    #     cutoff=[0.1689148094842286, 0.1960629961152209],
    # )

    # ---

    # TODO: only float and int values are supported. Int values must be manually converted within the optim. function.

    # optim_method = "lhs"  # "gridsearch", "lhs"; "basinhopping"  # DEFINED ABOVE
    optim_seed = 26552
    max_iters = 3
    i0_iter = 0  # Index of the first iteration

    # --- Starting list of input files. CL args will append to this one.
    input_files = list()
    #
    # -------------------
    # 2-SECTIONS R(T), wis-optimized, fas
    # # -()- Key West
    # input_files.append("input_params/wis_optimized/keywest-2016.in")
    # input_files.append("input_params/wis_optimized/keywest-2017.in")
    # input_files.append("input_params/wis_optimized/keywest-2018.in")

    # # -()- Los Angeles
    # input_files.append("input_params/wis_optimized/losangeles-2018.in")
    # input_files.append("input_params/wis_optimized/losangeles-2019.in")
    # input_files.append("input_params/wis_optimized/losangeles-2020.in")

    # # -()- Maricopa
    # input_files.append("input_params/wis_optimized/maricopa-2018.in")
    # input_files.append("input_params/wis_optimized/maricopa-2019.in")
    # input_files.append("input_params/wis_optimized/maricopa-2020.in")

    # # -()- Miami
    # input_files.append("input_params/wis_optimized/miami-2019.in")
    # input_files.append("input_params/wis_optimized/miami-2020.in")
    # input_files.append("input_params/wis_optimized/miami-2021.in")

    # ---

    # # -()- Mia-Wynwood
    # input_files.append("input_params/sublocs_dev/mia-wynwood_2019.in")
    # input_files.append("input_params/sublocs_dev/mia-wynwood_2020.in")
    # input_files.append("input_params/sublocs_dev/mia-wynwood_2021.in")

    # # -()- Mia-South Beach
    # input_files.append("input_params/sublocs_dev/mia-southbeach_2019.in")
    # input_files.append("input_params/sublocs_dev/mia-southbeach_2020.in")
    # input_files.append("input_params/sublocs_dev/mia-southbeach_2021.in")

    # # # ----      TRAPNIGHT – NO SEASONAL TREND
    # # -()- LA TRAPNIGHT
    # input_files.append("input_params/by_trapnight/no-season-trend/losangeles-2018.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/losangeles-2019.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/losangeles-2020.in")

    # # -()- Maricopa TRAPNIGHT
    # input_files.append("input_params/by_trapnight/no-season-trend/maricopa-2018.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/maricopa-2019.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/maricopa-2020.in")

    # # -()- Mia TRAPNIGHT
    # input_files.append("input_params/by_trapnight/no-season-trend/miami-2019.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/miami-2020.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/miami-2021.in")

    # # -()- Key West
    # input_files.append("input_params/by_trapnight/no-season-trend/keywest-2016.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/keywest-2017.in")
    # input_files.append("input_params/by_trapnight/no-season-trend/keywest-2018.in")

    # # # ----      TRAPNIGHT – **WITH** SEASONAL TREND
    # # -()- LA TRAPNIGHT
    # input_files.append("input_params/by_trapnight/losangeles-2018.in")
    # input_files.append("input_params/by_trapnight/losangeles-2019.in")
    # input_files.append("input_params/by_trapnight/losangeles-2020.in")

    # # -()- Maricopa TRAPNIGHT
    # input_files.append("input_params/by_trapnight/maricopa-2018.in")
    # input_files.append("input_params/by_trapnight/maricopa-2019.in")
    # input_files.append("input_params/by_trapnight/maricopa-2020.in")

    # -()- Mia TRAPNIGHT
    input_files.append("input_params/by_trapnight/miami-2019.in")
    input_files.append("input_params/by_trapnight/miami-2020.in")
    input_files.append("input_params/by_trapnight/miami-2021.in")

    # # -()- Key West
    # input_files.append("input_params/by_trapnight/keywest-2016.in")
    # input_files.append("input_params/by_trapnight/keywest-2017.in")
    # input_files.append("input_params/by_trapnight/keywest-2018.in")


    # ---------------------------------------


    # --- Scoring parameters
    week_weights = pd.Series(  # Defines the relative importance of each week.
        data=[1.0, 1.0, 0.75, 0.50],
        index=[1, 2, 3, 4],
        name="weeks_ahead",
    )

    # # -()- Weight of each score category
    # print(Fore.YELLOW + " = = = = WARNING - YOU'RE USING COVERAGE AS optim METRICS =======" + Style.RESET_ALL)
    # score_weights = dict()  # defaultdict(lambda: 1.0)  # Use default dict if you're confident with the names
    # score_weights["accuracy"] = 0.0
    # score_weights["sharpness"] = 0.0
    # score_weights["baseline-wis"] = 0.0
    # score_weights["coverage_50"] = 1.0
    # score_weights["coverage_95"] = 1.0
    # score_weights["noise_coef"] = 0.0
    #
    # # -()- Weight of each score category - WIS
    # score_weights = dict()  # defaultdict(lambda: 1.0)  # Use default dict if you're confident with the names
    # score_weights["accuracy"] = 0.0
    # score_weights["sharpness"] = 1.0
    # score_weights["baseline-wis"] = 1.0
    # score_weights["coverage_50"] = 0.0
    # score_weights["coverage_95"] = 0.0
    # score_weights["noise_coef"] = 0.0

    # # -()- Weight of each score category - BASELINE_RELATIVE WIS
    # print(Fore.YELLOW + " = = = = WARNING - YOU'RE USING RELATIVE WIS AS optim METRICS =======" + Style.RESET_ALL)
    # score_weights = dict()  # defaultdict(lambda: 1.0)  # Use default dict if you're confident with the names
    # score_weights["accuracy"] = 0.0
    # score_weights["sharpness"] = 0.0
    # score_weights["baseline-wis"] = 1.0
    # score_weights["coverage_50"] = 0.0
    # score_weights["coverage_95"] = 0.0
    # score_weights["noise_coef"] = 0.0

    # Noise coefficient punishment parameters (exp barrier)
    nc0 = 2.0  # Value around which the cost starts to increase relevantly
    nc_exp_gamma = 10.  # Sharpness of the exp increase
    nc_exp_delta = 0.3  # The cost raises ~10-fold ("very high") around nc0 + delta

    # Alpha-scores parameters (quadratic distance)
    # alpha_qwidth = 0.2  # REGULAR - Score is around 1 if the coverage is this far from target value
    alpha_qwidth = 0.1  # COMPARE W/ SHARPNESS - Score is around 1 if the coverage is this far from target value


def main():

    params = ProgramParams()
    args, other_args = parse_args(params)
    data = ProgramData()

    export_params_yaml(params, args, other_args)

    run_optimization(params, args, other_args, data)
    export_final_data(params, args, other_args, data)


class ProgramData:
    """Intermediary data containers and variables."""
    input_dicts: OrderedDict[str, dict]
    truth_data_dict = None
    param_names = None

    score_list_of_dicts = list()  # List of score tables: one for each iteration-input
    score_indexing_list = list()  # List of index elements for the summary score df

    param_samples: np.ndarray  # MultiIndex with all parameter sets used during the optimization
    final_score_sr: pd.Series  # A series with the final score of each iteration
    score_summary_df: pd.DataFrame  # Dataframe with scores divided by type and by input file index

    i_iter = 0


# --------------------------------------------------------------
# ==============================================================


def parse_args(params: ProgramParams):
    parser = ArgumentParser(
        description="Use it with exactly one `-o` argument, and as many `-i` arguments as needed."
    )

    parser.add_argument("-i", "--input-file", action="append", default=params.input_files)
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("-m", "--max-iters", type=int, default=None)
    # parser.add_argument("--optim-method")

    # Parse arguments
    args, other_args = parser.parse_known_args()

    # Check parsed args
    if args.output_dir is None:
        raise ArgumentTypeError("Hey, argument '-o, --output-dir'  is required. Please inform a "
                                "path to the output directory ")

    if args.input_file is None or len(args.input_file) == 0:
        raise ArgumentTypeError("Hey, the list of input files is empty. Please inform at least one "
                                "as '-i, --input-file'.")

    return args, other_args


def export_params_yaml(params: ProgramParams, args, other_args):
    """Dump program metadata to a yaml file"""
    class_dict = {key: value for key, value in params.__class__.__dict__.items() if not key.startswith('__') and not callable(value)}
    out_dict = copy.deepcopy(class_dict)

    # PROCESSINGS TO DO:
    # [ ] Convert ordered dict to regular dict
    # [ ] 'week_weights': convert to dict (.to_dict)

    # --- Ordered dict to regular dict
    out_dict["vary_params"] = dict(out_dict["vary_params"])
    out_dict["vary_params"] = dict(out_dict["vary_params"])

    # --- Series to dict
    out_dict["week_weights"] = out_dict["week_weights"].to_dict()
    out_dict["score_weights"] = dict(out_dict["score_weights"])

    # --- Parameters overriden by CL args
    out_dict["input_files"] = args.input_file

    # ETC: other conversions (by type)
    prepare_dict_for_yaml_export(out_dict)

    # output dir
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metadata.yaml"), "w") as fp:
        yaml.dump(out_dict, stream=fp)

    # Max iters
    if args.max_iters is not None:
        params.max_iters = args.max_iters


# ---------------------------------------------------------------------------
# =================================================================
# SCORING METHODS
# ---------------------------------------------------------------------------


def _aux_exp_barrier(x, x0, delta, gamma=10.):
    """Returns an exponential function of x that works as a cost barrier around x0.
    The cost is practically zero for x < x0.
    x0 is the "warning" value, where the cost is around 0.6
    at x0 + delta, the cost is around 12 (10-fold increase), considered as prohibitive.
    The parameter `gamma` adjusts the sharpness of the exponential increase.
    """
    amp = np.exp(2.5 - gamma * delta)    # exp(2.5) ~= 12
    return amp * np.exp(gamma * (x - x0))


def _aux_quad_dist(x, x0, width):
    """Simple quadaratic distance of x from target x0. The score is 1 if |x - x0| = width."""
    return (x - x0)**2 / width**2


def make_season_scoretable(sdo: SweepDatesOutput, params: ProgramParams, data: ProgramData):
    """Extract scalar metrics from a sweep run (e.g. an entire season).
    Return a series keyed by the name of each metric.
    """
    # Preliminary calculations
    # ------------------------
    score = OrderedDict()

    # Make a series of truth values that's aligned with the forecast results dataframe (q_df, score_df)
    truth_sr = sdo.mosq.df[sdo.species]
    truth_sr_alig = aux_make_truth_vals_aligned(sdo.score_df.index, truth_sr)

    # Make a series of horizon-weeks that's aligned with the forecast df
    week_weights_alig = params.week_weights.loc[sdo.score_df.index.get_level_values("weeks_ahead")]
    week_weights_alig.index = sdo.score_df.index

    # Make a per-date score table that's weighted by horizon weeks
    weighted_score_df = sdo.score_df.astype(float).multiply(week_weights_alig, axis=0)

    # Weight the mosquito abundance by horizon week to use as a denominator for abundance-scaled metrics, like WIS
    weighted_scale = (truth_sr_alig * week_weights_alig).sum()  # Denominator to normalize for mosquito abundance

    # Evaluative scores - aggregated by mean
    # -----------------
    # --- Accuracy
    score["accuracy"] = weighted_score_df["score_wis_calib"].sum() / weighted_scale

    # --- Sharpness: rewards smaller forecast quantiles
    score["sharpness"] = weighted_score_df["score_wis_sharp"].sum() / weighted_scale

    # # --- Alpha-coverage precision
    score["coverage_50"] = _aux_quad_dist(
        weighted_score_df["score_alpha_50"].sum() / week_weights_alig.sum(),
        0.50, params.alpha_qwidth
    )
    score["coverage_95"] = _aux_quad_dist(
        weighted_score_df["score_alpha_95"].sum() / week_weights_alig.sum(),
        0.95, params.alpha_qwidth
    )
    # score["coverage_50"] = sdo.score_df["score_alpha_50"]   # Use unweighted

    # --- Baseline-Relative WIS
    score["baseline-wis"] = weighted_score_df["score_rel_baseline_wis"].mean()

    # --- Consistency (how? Rel WIS in each date is not good).

    # Extra punishments and rewards - aggregated by adding
    # ------------------------------
    # --- Noise coefficient
    nc = float(sdo.input_dict["noise_coef"])
    score["noise_coef"] = _aux_exp_barrier(nc, params.nc0, params.nc_exp_delta, params.nc_exp_gamma)

    # FINAL SCORE meant to be used for optimization
    ev_values, ev_weights = np.array(
        [[score[n], params.score_weights[n]]
         for n in score.keys()
         # for n in ["accuracy", "sharpness", "baseline-wis", "coverage_50", "coverage_95"]
         ]).T

    score["TOTAL"] = (
            np.average(ev_values, weights=ev_weights)  # Evaluative scores
            # np.average([score[n] * params.score_weights[n]
            #            for n in ["accuracy", "sharpness", "coverage_50", "coverage_95"]])  # Evaluative
            + sum([score[n] * params.score_weights[n]
                  for n in ["noise_coef"]])  # Extra punishments/rewards
    )

    return score


def run_iteration(
        param_vals: np.ndarray, params: ProgramParams, args, other_args, data: ProgramData
):
    """ Run the forecast for all input files and calculates the overall score.
    This is the main function to be optimized in this script.
    """
    print(Style.BRIGHT + "====================================" + Style.RESET_ALL)
    print(f"--- START ITERATION {data.i_iter} (max = {params.max_iters}) @ {datetime.datetime.now()}")
    print(pd.Series(param_vals, index=data.param_names))

    num_inputs = len(data.input_dicts)
    # Set first iteration index
    if params.i0_iter and data.i_iter == 0:
        data.i_iter = params.i0_iter

    e_xt0 = time.time()

    def task(input_tup):

        # ----------------------------------------------------------------------------
        # FORECAST STAGE
        # ----------------------------------------------------------------------------
        i_input, input_dict = input_tup
        input_dict: dict

        mod_input_dict = input_dict.copy()

        # --- Replace parameters
        for name, val in zip(data.param_names, param_vals):
            # --- Interpret special parameters
            if name == "rt_width":
                assert(0 < val <= 1)
                mod_input_dict["q_low"] = (1. - val) / 2.
                mod_input_dict["q_hig"] = (1. + val) / 2.

            # Integer values
            if name in ["preproc_roi_len", "ndays_past"]:
                mod_input_dict[name] = str(int(val))  # Take the FLOOR

            else:  # --- Standard parameters
                mod_input_dict[name] = str(val)

        # Additional parametesr to the sweep_forecast_dates routine
        mod_input_dict["original_input_file"] = args.input_file[i_input]
        mod_input_dict["export_fore_quantiles"] = "True"
        mod_input_dict["export_preprocess"] = "False"
        mod_input_dict["export_rt_pastfore"] = "False"

        # --- Export temporary input file
        tmp_input_fname = TMP_INPUT_FMT.format(i_input, np.random.randint(1000000 - 1) + i_input)
        out_dir = os.path.join(args.output_dir, OUT_SUBDIR_FMT.format(f"step{data.i_iter:03d}_file{i_input:03d}"))

        try:  # Prevents tmp files from remaining in storage due to errors
            write_config_file(tmp_input_fname, mod_input_dict)

            # --- Count and run
            # print(f"Simulation {i_sim + 1} of {num_sim} (c = {cutoff:0.4f}, n = {noise_coef:0.4f})")
            cmd_list = f"{FORECAST_CALL} {tmp_input_fname} {out_dir}".split() + other_args
            cmd_list += "--no-plots".split()  # ADDITIONAL ARGUMENTS
            cmd_list += f"--ncpus {N_SWEEP_CPUS}".split()
            print(" ".join(cmd_list))  # Shows command being run

            try:
                # === CALL THE FORECAST SCRIPT
                call_result = subprocess.run(cmd_list, check=True)  # ==== CALL THE FORECAST
            except subprocess.CalledProcessError:
                print(Fore.RED + f"Problem on simulation which would export into `{out_dir}`. Breaking the loop..." + Style.RESET_ALL)
                raise

        finally:
            os.remove(tmp_input_fname)

        # ----------------------------------------------------------------------------
        # SCORING STAGE
        # ----------------------------------------------------------------------------
        sdo: SweepDatesOutput = load_sweep_forecast(out_dir)  # Takes less than 0.05 second, so it should be fine.
        sdo.score_df["score_wis_sharp"] = sdo.score_df["score_wis"] - sdo.score_df["score_wis_calib"]

        score_table = make_season_scoretable(sdo, params, data)

        print(Fore.YELLOW + f"ITER {data.i_iter} | FILE {i_input}: {score_table}" + Style.RESET_ALL)

        return score_table

    content = list(enumerate(data.input_dicts.values()))

    # --- RUN THE FORECASTS and SCORING
    score_tables = map_parallel_or_sequential(task, content, ncpus=N_INPUT_CPUS)

    # --- Build the scoring summary of this iteration
    data.score_list_of_dicts += score_tables
    data.score_indexing_list += [(*param_vals, i) for i in range(num_inputs)]  # Build items of the composite index

    e_xtf = time.time()
    print(f"Iteration {data.i_iter} execution time = {e_xtf - e_xt0:0.4f}s")
    print("\n===================================================\n")

    data.i_iter += 1

    # --- FINALLY: RETURN A SIMPLE AVERAGE OF THE TOTAL SCORES FOR EACH INPUT FILE
    return sum(score["TOTAL"] for score in score_tables) / len(score_tables)


# ------------------------------------------------------------------------------
# OPTIMIZING METHODS
# ------------------------------------------------------------------------------

def _aux_get_bounds(vary_params: OrderedDict):
    """Get maxium and minimum values from the list of param values."""
    upper_bounds = np.fromiter((max(vals) for vals in vary_params.values()), dtype=float)
    lower_bounds = np.fromiter((min(vals) for vals in vary_params.values()), dtype=float)

    return upper_bounds, lower_bounds


def opt_gridsearch(opt_func, params: ProgramParams, args, data: ProgramData):
    """Runs the model for all parameter combinations."""
    print("LOG: GRIDSEARCH METHOD START")

    nparams = len(params.vary_params)
    # Create a grid with all informed values
    grid = np.meshgrid(*list(params.vary_params.values()), indexing="ij")
    param_samples = np.stack(grid, axis=-1).reshape(-1, nparams)
    param_index = pd.MultiIndex.from_arrays(  # Sequential index with the parameter values
        param_samples.T, names=list(params.vary_params.keys()))

    print("PARAMETERS TO TEST")
    print(param_index)

    scores = np.apply_along_axis(opt_func, axis=1, arr=param_samples)

    # --- Build the result structures
    # Final scores series, keyed by parameters
    data.param_samples = param_samples
    data.final_score_sr = pd.Series(scores, index=param_index)
    data.final_score_sr.name = "score"


def opt_lhs(opt_func, params: ProgramParams, args, data: ProgramData):
    """"""
    print("LOG: LHS METHOD START")

    upper_bounds, lower_bounds = _aux_get_bounds(params.vary_params)

    # Generate a Latin hypercube sample - HAS SEED LEAP
    lhs = LatinHypercube(d=lower_bounds.shape[0], seed=params.optim_seed + params.i0_iter).random(params.max_iters)

    # Scale the Latin hypercube sample to the bounds of the search space
    param_samples = lower_bounds + (upper_bounds - lower_bounds) * lhs
    param_index = pd.MultiIndex.from_arrays(  # Sequential index with the parameter values
        param_samples.T, names=list(params.vary_params.keys()))

    print("PARAMETERS TO TEST")
    print(param_index)

    # Evaluate the function for each parameter set
    scores = np.apply_along_axis(opt_func, axis=1, arr=param_samples)  # Scores for all iterations

    # --- Build the result structures
    # Final scores series, keyed by parameters
    data.param_samples = param_samples
    data.final_score_sr = pd.Series(scores, index=param_index)
    data.final_score_sr.name = "score"


# OPTIMIZER SIGNATURE: f(opt_func, params, args, data)
opt_callable_dict = dict(
    gridsearch=opt_gridsearch,
    lhs=opt_lhs,
)


def run_optimization(params: ProgramParams, args, other_args, data: ProgramData):
    """"""
    # vals = [ ]
    data.input_dicts = OrderedDict((fname, read_config_file(fname)) for fname in args.input_file)
    #   ^  signature: a[fname] = input_dict
    data.param_names = list(params.vary_params.keys())

    # --- Define function to optimize
    def opt_func(vals):
        """Final callable to the optimizing process"""
        result = run_iteration(vals, params, args, other_args, data)

        return result  # TODO: rethink about this. Could be defined within each optimizer.

    # --- Call the optimizing routine
    opt_xt0 = time.time()
    opt_callable_dict[params.optim_method](opt_func, params, args, data)
    opt_xt = time.time() - opt_xt0

    print("======================================================")
    print(f"--- Total time spent in optimization: {opt_xt}s ({seconds_to_hhmmss(opt_xt)})")
    print(f"Finished at {datetime.datetime.now()}")

    # --- AGGREGATE RESULTS - build the scores by type and by input file
    data.score_summary_df = (
        pd.DataFrame(data.score_list_of_dicts,
                     index=pd.MultiIndex.from_tuples(
                         data.score_indexing_list,
                         names=[*data.param_names, "i_input"]
                     ))
    )

    # Visualize the results
    print(data.score_summary_df)


# =================================================


def export_final_data(params, args, other_args, data):

    # --- Detailed Summary File
    score_summary_fname = os.path.join(args.output_dir, "score_summary.csv")
    data.score_summary_df.to_csv(score_summary_fname)

    # --- Final scores file
    final_score_fname = os.path.join(args.output_dir, "final_scores.csv")
    data.final_score_sr.to_csv(final_score_fname)


if __name__ == "__main__":
    main()


# ---------------------------------
# =========================================================================
# ------------------------------------
# OLD INPUTS AND CODE AND ETC
# -
# ------------------------------------

    # # -()- Key West
    # input_files.append("input_params/optim/automatic_optim/keywest/keywest_2017.in")
    # input_files.append("input_params/optim/automatic_optim/keywest/keywest_2018.in")
    # input_files.append("input_params/optim/automatic_optim/keywest/keywest_2019.in")

    # # -()- Los Angeles
    # input_files.append("input_params/optim/automatic_optim/losangeles/losangeles_2018.in")
    # input_files.append("input_params/optim/automatic_optim/losangeles/losangeles_2019.in")
    # input_files.append("input_params/optim/automatic_optim/losangeles/losangeles_2020.in")

    # # -()- Maricopa
    # input_files.append("input_params/optim/automatic_optim/maricopa/maricopa_2018.in")
    # input_files.append("input_params/optim/automatic_optim/maricopa/maricopa_2019.in")
    # input_files.append("input_params/optim/automatic_optim/maricopa/maricopa_2020.in")
    # # input_files.append("input_params/optim/automatic_optim/maricopa/maricopa_###202###1.in") # -- Ditched

    # # -()- Miami 2019-2020-2021
    # input_files.append("input_params/optim/automatic_optim/miami/miami_2019.in")
    # input_files.append("input_params/optim/automatic_optim/miami/miami_2020.in")
    # input_files.append("input_params/optim/automatic_optim/miami/miami_2021.in")



# -------------------
# 2-SECTIONS R(T), wis-optimized, fas
# # -()- Key West
# input_files.append("input_params/wis_optimized/keywest-2016.in")
# input_files.append("input_params/wis_optimized/keywest-2017.in")
# input_files.append("input_params/wis_optimized/keywest-2018.in")

# # -()- Los Angeles
# input_files.append("input_params/wis_optimized/losangeles-2018.in")
# input_files.append("input_params/wis_optimized/losangeles-2019.in")
# input_files.append("input_params/wis_optimized/losangeles-2020.in")

# # -()- Maricopa
# input_files.append("input_params/wis_optimized/maricopa-2018.in")
# input_files.append("input_params/wis_optimized/maricopa-2019.in")
# input_files.append("input_params/wis_optimized/maricopa-2020.in")

# # -()- Miami
# input_files.append("input_params/wis_optimized/miami-2019.in")
# input_files.append("input_params/wis_optimized/miami-2020.in")
# input_files.append("input_params/wis_optimized/miami-2021.in")


# --------------------
# SEQUENTIAL R(t)
# # -()- Key West
# input_files.append("input_params/seq-rt_in/keywest-2012.in")
# input_files.append("input_params/seq-rt_in/keywest-2013.in")
# input_files.append("input_params/seq-rt_in/keywest-2014.in")

# # -()- Los Angeles
# input_files.append("input_params/seq-rt_in/losangeles-2019.in")
# input_files.append("input_params/seq-rt_in/losangeles-2020.in")

# # -()- Maricopa - TODO change for sequential R(t)
# input_files.append("input_params/wis_optimized/maricopa-2018.in")
# input_files.append("input_params/wis_optimized/maricopa-2019.in")
# input_files.append("input_params/wis_optimized/maricopa-2020.in")

# # -()- Miami
# input_files.append("input_params/seq-rt_in/miami-2020.in")
# input_files.append("input_params/seq-rt_in/miami-2021.in")

