"""
Import and export files fot the CDC Flu Forecast Challenge
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import os
import pandas as pd
import warnings

from collections import OrderedDict, defaultdict
from collections.abc import Sequence

import yaml

import rtrend_tools.utils as utils

from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES, NUM_STATES, WEEKDAY_TGT, \
    NUM_OUTP_LINES, NUM_OUTP_LINES_WPOINTS, get_next_weekstart
from rtrend_tools.forecast_structs import MosqData, TgData, CDCDataBunch, ForecastPost, SweepDatesOutput
from toolbox.file_tools import write_config_string, read_file_header, read_config_file, str_to_bool_safe, HEADER_END, \
    read_config_strlist


# --------------------
# AUX

def read_yaml_simple(fname: str) -> dict:
    with open(fname, "r") as fp:
        d = yaml.load(fp, yaml.Loader)
    return d


def prepare_dict_for_yaml_export(d: dict):
    """Converts some data types within a dictionary into other objects
    that can be read in a file (e.g. strings).
    Operates recursively through contained dictionaries.
    Changes are made inplace for all dictionaries.
    """
    for key, val in d.items():

        # Recurse through inner dictionary
        if isinstance(val, dict):
            prepare_dict_for_yaml_export(val)

        # pathlib.Path into its string
        if isinstance(val, Path):
            d[key] = str(val.expanduser())

        # Timestamps into string repr.
        if isinstance(val, pd.Timestamp):
            d[key] = str(val)

# ---------------------


def load_cdc_truth_data(fname):
    """Load a generic truth data file from CDC."""
    cdc = CDCDataBunch()

    # Import
    cdc.df = pd.read_csv(fname, index_col=(0, 1), parse_dates=["date"])

    # Extract unique names and their ids
    cdc.loc_names = np.sort(cdc.df["location_name"].unique())  # Alphabetic order
    cdc.num_locs = cdc.loc_names.shape[0]
    cdc.loc_ids = cdc.df.index.levels[1].unique()

    # Make location id / name conversion
    cdc.to_loc_name = dict()  # Converts location id into name
    cdc.to_loc_id = dict()  # Converts location name into id
    for l_id in cdc.loc_ids:
        name = cdc.df.xs(l_id, level=1).iloc[0]["location_name"]  # Get the name from the first id occurrence
        cdc.to_loc_name[l_id] = name
        cdc.to_loc_id[name] = l_id

    cdc.df.sort_index(level="date", inplace=True)  # Sort by dates.

    cdc.data_time_labels = cdc.df.index.levels[0].unique().sort_values()

    return cdc


def week_str_to_weekstart_label(s):
    """
    Converts a string given as 'YYYY-WW' into a pandas timestamp referring to the corresponding weekstart label.

    ALGORITHM:
    ---------
    Take first day of the given year.
    Add the specified number of weeks.
    Take the next weekstart (Sunday 0am) label.

    For the dataset from 2019 to 2021, we did not have conflicts in the edge between years.
    """
    yw = s.split("-")
    y = int(yw[0].strip())
    w = int(yw[1].strip()) - 1  # TODO: subtract one, cause it starts at 1!!!!!!!

    return get_next_weekstart(datetime(y, 1, 1) + timedelta(weeks=w))


def load_mosquito_test_data(fname, date_col="Date"):
    """Loads the test mosquito dataset made by Andre."""

    mosq = MosqData()

    # Read the df
    mosq.df = pd.read_csv(fname, header=0, na_values=["", "ND"])

    # Set the index as the weekstart labels
    mosq.data_time_labels = mosq.df.index = \
        mosq.df[date_col].apply(week_str_to_weekstart_label, convert_dtype=pd.Timestamp)

    # --- Mosquito species data
    mosq.species_names = [name for name in mosq.df.columns if name not in [date_col, "id", "OID"]]
    mosq.num_species = len(mosq.species_names)

    return mosq


def remove_duplicated_mosq_entries_inplace(mosq: MosqData):
    """
    Just drops entries for which an equal index have already appeared in the mosq.df dataframe.
    Renews also the mosq.data_time_labels
    """
    dup = mosq.df.index.duplicated()
    mosq.df = mosq.df.loc[~dup]
    mosq.data_time_labels = mosq.df.index


def load_tg_dynamic_data(fname, tg_max=None):
    tg = TgData()
    tg.df = pd.read_csv(fname, index_col=0, parse_dates=[0])

    if tg_max is None:
        # -()- Calculate truncation point using mean and variance (better method: use quantiles)
        mean = (tg.df["Shape"] / tg.df["Rate"]).array
        std = np.sqrt(tg.df["Shape"] / tg.df["Rate"]**2)
        trunc_array = mean + 5 * std  # Criterion of truncation for each gamma distribution
        tg.max = int(np.ceil(trunc_array.max()))  # Takes the maximum

    else:
        # Fixed value
        tg.max = tg_max

    return tg


def export_forecast_simple(fname, post: ForecastPost):
    """Export one-species forecast into a simple csv file."""
    data = {q: vals for q, vals in zip(post.quantile_seq, post.weekly_quantiles)}
    df = pd.DataFrame(data, index=post.fore_time_labels)
    df.index.name = "Date"
    df.columns.name = "Quantiles"

    utils.make_dir_from(fname)
    df.to_csv(fname)


def export_forecast_extra_info(fname, post: ForecastPost, mosq: MosqData):
    """Exports surrounding data from a one-species forecast, like factual past data and preprocessed data."""
    factual_ct: pd.Series = mosq.df.loc[post.day_0:post.day_fore][post.species].rename("data")

    out_df = pd.DataFrame(post.float_data_daily, columns=["data_preproc"], index=post.past_daily_tlabels)
    out_df = pd.concat([out_df, factual_ct], axis=1)  # All Will be converted to float
    out_df.index.name = "Date"

    utils.make_dir_from(fname)
    out_df.to_csv(fname)


def export_forecast_cdc(fname, post_list, us, cdc, nweeks_fore, use_as_point=None,
                        add_week_to_labels=False):
    """
    Data is not assumed to strictly follow the CDC guidelines (all locations and quantiles),
    but warnings are thrown for irregular data.
    """

    # PPREAMBLE
    # ------------------------------------------------------------------------------------------------------------------
    today = datetime.today().date()
    today_str = today.isoformat()
    target_fmt = "{:d} wk ahead inc flu hosp"

    valid_post_list = [post for post in post_list if post is not None]
    num_valid = len(valid_post_list)

    # Check output size
    if num_valid != NUM_STATES:
        warnings.warn(f"\n[CDC-WARN] Hey, the number of calculated states/jurisdictions ({num_valid} "
                      f"+ US) does not match that required by CDC ({NUM_STATES} + US).\n")

    if use_as_point is not None:
        warnings.warn("HEY, use_as_point is not implemented in export_forecast_cdc.")

    # --- Build list of data arrays to be concatenated in the end
    # forecast_date_list =  # Date of the forecast. Same for entire doc, created later.
    location_list = list()  # Location (state) id
    target_list = list()  # String describing the week of forecast
    target_end_date_list = list()  # Date of the forecast report
    type_list = list()  # Type of output: "point" or "quantile"
    quantile_list = list()  # q-value of the quantile
    value_list = list()  # Value of the forecast data.
    actual_num_outp_lines = 0

    # DATA UNPACKING AND COLLECTION
    # ------------------------------------------------------------------------------------------------------------------

    # Definition of the processing routine
    # ------------------------------------
    def process_location(weekly_quantiles, num_q, quantile_seq, fore_time_labels, state_name):
        num_lines = nweeks_fore * (num_q + int(use_as_point is not None))  # Add point line, if requested

        # CDC format compliance check
        for date in fore_time_labels:
            if date.weekday() != WEEKDAY_TGT:
                warnings.warn(f"\n[CDC-WARN] Hey, wrong weekday ({date.weekday()}) was found in "
                              f"forecast data labels!)\n")

        # Allocate arrays
        # forecast_date_array = not needed here
        location_array = np.repeat(cdc.to_loc_id[state_name], num_lines)
        target_array = np.empty(num_lines, dtype=object)
        target_end_date_array = np.empty(num_lines, dtype=object)
        type_array = np.empty(num_lines, dtype=object)
        quantile_array = np.empty(num_lines, dtype=float)
        value_array = np.empty(num_lines, dtype=float)

        # ---
        i_line = 0
        for i_week in range(nweeks_fore):  # Loop over forecast weeks
            week = fore_time_labels[i_week] + int(add_week_to_labels) * timedelta(7)

            # Write data into arrays
            target_array[i_line:i_line + num_q] = target_fmt.format(i_week + 1)
            target_end_date_array[i_line: i_line + num_q] = week.date().isoformat()
            type_array[i_line: i_line + num_q] = "quantile"
            quantile_array[i_line: i_line + num_q] = quantile_seq[:]
            value_array[i_line: i_line + num_q] = weekly_quantiles[:, i_week]

            i_line += num_q

            # Point
            if use_as_point is not None:
                pass  # NOT IMPLEMENTED
                i_line += 1

        # Store arrays
        # forecast_date_list  # Not needed
        location_list.append(location_array)
        target_list.append(target_array)
        target_end_date_list.append(target_end_date_array)
        type_list.append(type_array)
        quantile_list.append(quantile_array)
        value_list.append(value_array)

        return num_lines

    # Application for all states and the US
    # -------------------------------------
    for i_post, post in enumerate(valid_post_list):   # Loop over each location
        actual_num_outp_lines += process_location(post.weekly_quantiles, post.num_quantiles,
                                                  post.quantile_seq, post.fore_time_labels, post.species)

    # --- APPLY SEPARATELY FOR US
    # post_samp = valid_post_list[0]  # First as example, they better be the same!
    actual_num_outp_lines += process_location(us.weekly_quantiles, NUM_QUANTILES, CDC_QUANTILES_SEQ,
                                              us.fore_time_labels, "US")

    # Concatenate all arrays in desired order
    forecast_date = np.repeat(today_str, actual_num_outp_lines)
    location = np.concatenate(location_list)
    target = np.concatenate(target_list)
    target_end_date = np.concatenate(target_end_date_list)
    type_ = np.concatenate(type_list)
    quantile = np.concatenate(quantile_list)
    value = np.concatenate(value_list)

    # Final check on the number of lines in output
    expec_num_outp_lines = NUM_OUTP_LINES if use_as_point is None else NUM_OUTP_LINES_WPOINTS
    if actual_num_outp_lines != expec_num_outp_lines:
        warnings.warn("\n[CDC-WARN] Hey, total number of lines in file doesn't match the expected "
                      "value.\n"
                      f"Total = {actual_num_outp_lines}   |   Expected = {NUM_OUTP_LINES}")

    # DF CONSTRUCTION AND EXPORT
    # ------------------------------------------------------------------------------------------------------------------

    out_df = pd.DataFrame(OrderedDict(
        forecast_date=forecast_date,
        location=location,
        target=target,
        target_end_date=target_end_date,
        type=type_,
        quantile=quantile,
        value=value,
    ))

    print(out_df)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    out_df.to_csv(fname, index=False)


def export_sweep_forecast(fname, quantiles_df: pd.DataFrame, input_dict):
    """Export the forecast quantiles for the sweep_forecast_dates script."""
    utils.make_dir_from(fname)

    header = str(pd.Timestamp.today()) + '\n'

    # Write all content to buffers
    header += write_config_string(input_dict)
    content = quantiles_df.to_csv()

    with open(fname, "w") as fp:
        fp.write(header)
        fp.write(HEADER_END)
        fp.write(content)


def export_sweep_preprocess(fname, post_array: Sequence[ForecastPost]):

    # AGGREGATION
    # -----------------------------------
    denoised_items = list()  # Will contain pd Series with post.denoised_weekly
    floatdata_items = list()  # post.float_data_daily
    ctpast_items = list()
    daysback_items = list()

    for post in post_array:
        # --- Multiindex counterparts of the date labels indexes
        weekly_idx = \
            pd.MultiIndex.from_product([[post.day_pres], post.denoised_weekly.index], names=["day_pres", "day_past"])
        daily_idx = \
            pd.MultiIndex.from_product([[post.day_pres], post.past_daily_tlabels], names=["day_pres", "day_past"])

        # --- Denoised weekly
        denoised_items.append(pd.Series(post.denoised_weekly.values, index=weekly_idx, name="denoised_weekly"))
        # --- Float data daily
        floatdata_items.append(pd.Series(post.float_data_daily, index=daily_idx, name="float_data_daily"))
        # --- Integer, fully preprocessed data
        ctpast_items.append(pd.Series(post.ct_past, index=daily_idx, name="ct_past"))
        # # --- Aux: number of days before day_pres
        # daysback_items.append(pd.Series((post.day_pres - post.past_daily_tlabels).days,
        #                                 index=daily_idx, name="days_back"))

    # Final df aggregation (by lines, then by columns)
    out_df = pd.concat([
            # pd.concat(daysback_items),
            pd.concat(denoised_items),  # Each of these concatenate over lines
            pd.concat(floatdata_items),
            pd.concat(ctpast_items),
        ], axis=1)

    # # Include a column with the number of days past before day_pres
    # out_df.insert(0, "days_back", out_df.index.map(lambda x: (x[0] - x[1]).days))

    # EXPORT
    # -----------------------------------
    out_df.to_csv(fname)


def export_sweep_rt(rpast_fname, rfore_fname, post_varray: Sequence[ForecastPost], roi_len, export_past=True):
    """"""
    # RT Past
    # -------------------
    if not any([post.rt_past_median is None for post in post_varray]) and export_past:  # Checks if there are RT past objects to work

        rtpast_d = defaultdict(lambda: list())

        for post in post_varray:

            index = pd.MultiIndex.from_product([[post.day_pres], post.past_daily_tlabels[-roi_len:]],
                                               names=["day_pres", "day_past"])
            rtpast_d["r_median"].append(pd.Series(post.rt_past_median, index=index))
            rtpast_d["r_loquant"].append(pd.Series(post.rt_past_loquant, index=index))
            rtpast_d["r_hiquant"].append(pd.Series(post.rt_past_hiquant, index=index))

        # Concatenate each forecast into a single series
        concat_d = {key: pd.concat(series) for key, series in rtpast_d.items()}
        rtpast_df = pd.DataFrame(concat_d)

        # Export
        rtpast_df.to_csv(rpast_fname)

    # RT Future (Fore)
    # -------------------
    if not any([post.rt_fore_median is None for post in post_varray]):  # Checks if there are RT past objects to work

        rtfore_d = defaultdict(lambda: list())

        for post in post_varray:
            index = pd.MultiIndex.from_product([[post.day_pres], post.fore_daily_tlabels],
                                               names=["day_pres", "day_fore"])
            rtfore_d["r_median"].append(pd.Series(post.rt_fore_median, index=index))
            rtfore_d["r_loquant"].append(pd.Series(post.rt_fore_loquant, index=index))
            rtfore_d["r_hiquant"].append(pd.Series(post.rt_fore_hiquant, index=index))

        # Concatenate each forecast into a single series
        concat_d = {key: pd.concat(series) for key, series in rtfore_d.items()}
        rtfore_df = pd.DataFrame(concat_d)

        # Export
        rtfore_df.to_csv(rfore_fname)


def export_sweep_noise_params(fname, post_varray: np.ndarray):
    """"""
    out_df = pd.DataFrame({
        "day_pres": np.fromiter((post.day_pres for post in post_varray), dtype=pd.Timestamp),
        "std": np.fromiter((post.noise_obj.std for post in post_varray), dtype=float),
    },
        # index_label="day_pres",  # Not supported in this pandas version
    )
    out_df.set_index("day_pres", inplace=True)

    out_df.to_csv(fname)


def load_sweep_forecast(sim_dir, read_truth=True):
    """Imports data from a file produced by the sweep_forecast_dates script."""
    sdo = SweepDatesOutput()
    q_fname = os.path.join(sim_dir, "fore_quantiles.csv")
    p_fname = os.path.join(sim_dir, "preprocess.csv")
    rpast_fname = os.path.join(sim_dir, "rt_past.csv")
    rfore_fname = os.path.join(sim_dir, "rt_fore.csv")

    # Read main file contents
    header_lines = read_file_header(q_fname)
    sdo.input_dict = read_config_strlist(header_lines)
    sdo.q_df = pd.read_csv(q_fname, index_col=[0, 1], header=0, skiprows=len(header_lines)+1, parse_dates=[0])
    sdo.dates_ens = sdo.q_df.index.unique(level="day_pres")

    # Interpret entries of the header
    sdo.apply_scores = str_to_bool_safe(sdo.input_dict.get("apply_scores", "false"))
    sdo.species = sdo.input_dict["species"]
    sdo.nweeks_fore = int(sdo.input_dict["nweeks_fore"])

    if sdo.apply_scores:
        # Pick and separate columns with scoring info
        score_labels = [k for k in sdo.q_df.columns if "score_" in k]  # Select columns with scores
        sdo.score_df = sdo.q_df[score_labels].copy()
        sdo.q_df.drop(score_labels, axis=1, inplace=True)

    # Try to retrieve the ground truth values directly from the output file
    if "truth_vals" in sdo.q_df:
        sdo.truth_vals = sdo.q_df["truth_vals"]
        sdo.q_df.drop("truth_vals", axis=1, inplace=True)

    # Make q_df columns float values
    sdo.q_df.columns = sdo.q_df.columns.astype(float)

    # Tries to read truth data
    if read_truth:
        try:
            sdo.mosq = load_mosquito_test_data(sdo.input_dict["truth_data_fname"])
        except FileNotFoundError:
            sys.stderr.write(f"Warning: truth data file {sdo.input_dict['truth_data_fname']} not found.\n")
            sdo.mosq = "FileNotFound"

    # Tries to read a file with preprocessing data
    if os.path.exists(p_fname):
        sdo.prep_df = pd.read_csv(p_fname, index_col=[0, 1], header=0, parse_dates=[0, 1])
        # Make a special column with number of days until day_pres
        sdo.prep_df.insert(0, "days_back", sdo.prep_df.index.map(lambda x: (x[0] - x[1]).days))

    # Tries to import R(t) (past and fore) data frames
    for fname, attrname in [(rpast_fname, "rpast_df"), (rfore_fname, "rfore_df")]:
        if os.path.exists(fname):
            sdo.__setattr__(attrname, pd.read_csv(fname, index_col=[0, 1], header=0, parse_dates=[0, 1]))

    # Some preprocessing
    # ------------------
    if read_truth:

        if isinstance(sdo.mosq, MosqData):
            # --- Create a series with truth values aligned with sdo.q_df
            sdo.species_series = sdo.mosq.df[sdo.species]
            if not isinstance(sdo.truth_vals, pd.Series):  # Truth series not found in the main output file (fore_quantiles.csv)
                sdo.truth_vals = aux_make_truth_vals_aligned(sdo.score_df.index, sdo.species_series)

        # --- Calculate some derived scores
        if sdo.apply_scores:
            if "score_rel_wis" not in sdo.score_df and "score_wis" in sdo.score_df:
                sdo.score_df["score_rel_wis"] = sdo.score_df["score_wis"] / sdo.truth_vals

            if "score_wis_sharp" not in sdo.score_df and "score_wis_calib" in sdo.score_df:
                sdo.score_df["score_wis_sharp"] = sdo.score_df["score_wis"] - sdo.score_df["score_wis_calib"]

    else:  # truth_vals can't be retrieved
        pass

    return sdo


def convert_dtwk_to_foredate(dtwk: tuple):
    """
    Converts a tuple in with shape:
    (date: pd.Timestamp, nweeks_ahead: int)

    into the fore time labels, where:
    fore_t_label = date + nweeks_ahead(in weeks)

    To apply this to an index or array of tuples, use 'map'.
    """
    date, wk_ahead = dtwk
    return date + pd.Timedelta(wk_ahead, "w")


def aux_make_truth_vals_aligned(fore_mult_index, species_series):
    """
    Constructs a series of truth values aligned with the multiindex quantiles forecast dictionary.

    sdo = SweepDatesOut
    fore_mult_index = sdo.score_df.index
    species_series = sdo.mosq.df[sdo.species]`
    """

    def _convert(in_vals):
        """Converts present date and week ahead into actual forecast date."""
        date, wk_ahead = in_vals
        return date + pd.Timedelta(wk_ahead, "w")
    # lambda vals: vals[0] + pd.Timedelta(vals[1], "w")  # Lambda version

    fore_dt_index = fore_mult_index.map(_convert)

    # -()- CHOSE WHICHEVER LINE BELOW WORKS FOR THE LOCATION
    try:
        truth_vals = species_series.loc[fore_dt_index]  # Truth values aligned with quantiles df; KeyError if missing date
    except KeyError:
        truth_vals = species_series.reindex(fore_dt_index)  # Truth values aligned with quantiles df.

    truth_vals.index = fore_mult_index

    return truth_vals


def load_forecast_cdc(fname):
    """Import forecast file in the CDC format, possibly created with 'export_forecast_cdc()'. """

    df = pd.read_csv(fname, header=0, parse_dates=["forecast_date", "target_end_date"])

    return df


def make_state_arrays(weekly_quantiles, state_id, num_quantiles, nweeks_fore):
    pass
