"""
Plot the estimated R(t) over multiple years.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

import rtrend_tools.data_io as dio
import rtrend_tools.interpolate as interp
import rtrend_tools.preprocessing as preproc
import rtrend_tools.rt_estimation as mcmcrt
import rtrend_tools.visualization as vis

mpl.use('MacOSX')


def main():
    # General parameters
    species = "Ae_aegypti"
    filter_cutoff = 0.15
    filter_cutoff_str = f"{filter_cutoff:0.2f}".replace(".", "p")
    tg_max = 45
    export_preprocessed = None  # OVERRIDEN BELOW. If not overriden, the preprocessed mosquito series is not exported.
    do_calculate_mcmc = True  # If False, exits the code before calculating MCMC
    rerun_mcmc = True
    extend_weeks_left = 2  # 4  # Append data to the beginning of the time series, pushing the transient backwards
    remove_past_years = True  # Filters years before the first in list. Avoids trouble from some locations.

    # # -()- KEY WEST - Entire season (THAT TAKES TIME TO RUN!)
    # mosq_fname = "mosquito_data/Key_West_week.csv"
    # tg_data_fname = "tg_data/denoised/Key_West-Ae_aegypti_cutoff0p100.csv"
    # mcmc_out_pref = f"mcmc_outputs/Key_West/Ae_aegypti_tgc0p100_2011-2020_ctc{filter_cutoff_str}_".replace(".", "p")
    # export_preprocessed = f"mosquito_data/preprocessed/keywest/Ae_aegypti_tgc0p100_2011-to-2020_ctc{filter_cutoff_str}.csv"
    # years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

    # # -()- LOS ANGELES - Entire season
    # mosq_fname = "mosquito_data/LA_week.csv"
    # tg_data_fname = "tg_data/denoised/Los_Angeles-Ae_aegypti_cutoff0p100.csv"  # Lowpass filtered Tg
    # mcmc_out_pref = f"mcmc_outputs/Los_Angeles/Ae_aegypti_tgc0p100_2017-to-2021_ctc{filter_cutoff_str}_"
    # export_preprocessed = f"mosquito_data/preprocessed/losangeles/Ae_aegypti_tgc0p100_2017-to-2021_ctc{filter_cutoff_str}.csv"
    # years = [2017, 2018, 2019, 2020, 2021]

    # -()- MIAMI - Entire season
    mosq_fname = "mosquito_data/Miami_week.csv"
    tg_data_fname = "tg_data/denoised/Miami-Ae_aegypti_cutoff0p100.csv"  # Lowpass filtered Tg
    mcmc_out_pref = f"mcmc_outputs/Miami/Ae_aegypti_tgc0p100_2019-2022_ctc{filter_cutoff_str}_"
    export_preprocessed = f"mosquito_data/preprocessed/miami/Ae_aegypti_tgc0p100_2019-to-2022_ctc{filter_cutoff_str}.csv"
    years = [2019, 2020, 2021, 2022]

    # # -()- PHOENIX/MARICOPA - Entire season (long one!)
    # mosq_fname = "mosquito_data/Maricopa_week.csv"
    # tg_data_fname = "tg_data/denoised/Phoenix-Ae_aegypti_cutoff0p100.csv"
    # mcmc_out_pref = f"mcmc_outputs/Maricopa/Ae_aegypti_tgc0p100_2014-to-2021_ctc{filter_cutoff_str}_"
    # export_preprocessed = f"mosquito_data/preprocessed/maricopa/Ae_aegypti_tgc0p100_2014-to-2021_ctc{filter_cutoff_str}.csv"
    # years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

    # # -()- PHOENIX/MARICOPA - 1st section plus
    # mosq_fname = "mosquito_data/Maricopa_week.csv"
    # tg_data_fname = "tg_data/denoised/Phoenix-Ae_aegypti_cutoff0p100.csv"
    # mcmc_out_pref = f"mcmc_outputs/Maricopa/Ae_aegypti_tgc0p100_2014-to-2018_ctc{filter_cutoff_str}_"
    # export_preprocessed = f"mosquito_data/preprocessed/maricopa/Ae_aegypti_tgc0p100_2014-to-2018_ctc{filter_cutoff_str}.csv"
    # years = [2014, 2015, 2016, 2017, 2018]

    # # -()- PHOENIX/MARICOPA - 2nd section plus
    # mosq_fname = "mosquito_data/Maricopa_week.csv"
    # tg_data_fname = "tg_data/denoised/Phoenix-Ae_aegypti_cutoff0p100.csv"
    # mcmc_out_pref = f"mcmc_outputs/Maricopa/Ae_aegypti_tgc0p100_2018-to-2021_ctc{filter_cutoff_str}_".replace(".", "p")
    # export_preprocessed = f"mosquito_data/preprocessed/maricopa/Ae_aegypti_tgc0p100_2018-to-2021_ctc{filter_cutoff_str}.csv"
    # years = [2018, 2019, 2020, 2021]

    # # -()- BROWNSVILLE (NEEDS REFURBISH)
    # mosq_fname = "mosquito_data/Brownsville_week.csv"
    # tg_data_fname = "tg_data/denoised/Brownsville-Ae_aegypti_cutoff0p100.csv"
    # mcmc_out_pref = f"mcmc_outputs/Brownsville/Ae_aegypti_tgc0p100_2017-2018_ctc{filter_cutoff:0.2f}_".replace(".", "p")
    # years = [2017, 2018]

    # # -()- MIA-WYNWOOD
    # mosq_fname = "mosquito_data/Mia-Wynwood_week.csv"
    # tg_data_fname = "tg_data/denoised/Miami-Ae_aegypti_cutoff0p100.csv"  # Lowpass filtered Tg
    # mcmc_out_pref = f"mcmc_outputs/Mia-Wynwood/Ae_aegypti_tgc0p100_2019-2022_ctc{filter_cutoff_str}_"
    # export_preprocessed = f"mosquito_data/preprocessed/mia-wynwood/Ae_aegypti_tgc0p100_2019-to-2022_ctc{filter_cutoff_str}.csv"
    # years = [2019, 2020, 2021, 2022]
    # # # -//-

    # # -()- MIA-WYNWOOD
    # mosq_fname = "mosquito_data/Mia-Southbeach_week.csv"
    # tg_data_fname = "tg_data/denoised/Miami-Ae_aegypti_cutoff0p100.csv"  # Lowpass filtered Tg
    # mcmc_out_pref = f"mcmc_outputs/Mia-Southbeach/Ae_aegypti_tgc0p100_2019-2022_ctc{filter_cutoff_str}_"
    # export_preprocessed = f"mosquito_data/preprocessed/mia-southbeach/Ae_aegypti_tgc0p100_2019-to-2022_ctc{filter_cutoff_str}.csv"
    # years = [2019, 2020, 2021, 2022]
    # # # -//-

    # # -()- MIAMI - TESTS FOR THE EGG-LAYING MOSQUITO MODEL
    # mosq_fname = "mosquito_data/Miami_week.csv"
    # tg_data_fname = f"tg_data/denoised/Miami-Ae_aegypti_cutoff{filter_cutoff_str}0.csv"  # Lowpass filtered Tg
    # mcmc_out_pref = f"mcmc_outputs/Miami/Ae_aegypti_tgc0p100_2019-2022_ctc{filter_cutoff_str}_"
    # export_preprocessed = f"mosquito_data/preprocessed/miami/Ae_aegypti_tgc{filter_cutoff_str}0_2019-to-2022_ctc{filter_cutoff_str}.csv"
    # years = [2019]# , 2020, 2021, 2022]

    # # -()- MIAMI - TESTS FOR THE EGG-LAYING MOSQUITO MODEL
    # mosq_fname = "mosquito_data/Miami_week_dorian.csv"
    # tg_data_fname = f"tg_data/denoised/Miami-Ae_aegypti_cutoff{filter_cutoff_str}0.csv"  # Lowpass filtered Tg
    # mcmc_out_pref = f"mcmc_outputs/Miami/Ae_aegypti_tgc0p100_2019-2022_ctc{filter_cutoff_str}_TEST_"
    # export_preprocessed = f"mosquito_data/preprocessed/miami/Ae_aegypti_tgc{filter_cutoff_str}0_2019-to-2022_ctc{filter_cutoff_str}_TEST.csv"
    # years = [2019]  # , 2020, 2021, 2022]

    # ---------------------
    # Loading
    # -------------------------------------------------------------------------------
    mosq = dio.load_mosquito_test_data(mosq_fname)
    tg = dio.load_tg_dynamic_data(tg_data_fname, tg_max)

    # Count of days starts at the first day in TG file
    i_t_daily = np.arange(tg.df.shape[0])
    day_0 = tg.df.index[0]

    # Mosq data preprocess
    # ----------------------------------------------------------------

    # Select data
    series: pd.Series = mosq.df[species]
    series = series.loc[~series.index.duplicated(keep="last")]  # Remove duplicate dates (between years)
    # Remove trailing years
    series = series.loc[:pd.Timestamp(f"{max(years)}-12-31")]  # Remove data from future years
    if remove_past_years:
        series = series.loc[pd.Timestamp(f"{min(years)}-01-01"):]  # Remove data from past years (avoids some trouble)
    mosq_day_0 = series.index[0]
    i_t_weekly: np.ndarray = (series.index - mosq_day_0).days  # Array of integers representing day indexes in mosq data

    if extend_weeks_left:  # Fix the interpolation transient by adding data to the start of the series
        extend_days_left = extend_weeks_left * interp.WEEKLEN  # Convert to days
        # Concatenate int time stamps
        x = np.arange(i_t_weekly[0] - extend_days_left, i_t_weekly[0], interp.WEEKLEN)
        i_t_weekly = np.concatenate([x, i_t_weekly])
        i_t_weekly += extend_days_left  # Surprisingly, the spline fails if a value is negative!!
        # mosq_day_0 += pd.Timedelta(+i_t_weekly[0], "d")  # [WRONG, APAGAR] Compensates day_0 to keep sync
        mosq_day_0 -= pd.Timedelta(extend_days_left, "d")  # Compensates day_0 to keep sync

        # Concatenate mosquito data
        m_val = series[0]
        left_index = series.index[0] - np.arange(extend_weeks_left, 0, -1) * pd.Timedelta(1, "w")
        left_series = pd.Series(np.repeat(m_val, extend_weeks_left), index=left_index)
        series = pd.concat([left_series, series])

    # Denoise/interpolation
    denoised_float = preproc.apply_lowpass_filter_pdseries(series, cutoff=filter_cutoff).astype(np.longdouble)
    denoised = preproc.regularize_negatives_from_denoise(series, denoised_float, inplace=False)

    # -()- Regular interpolation
    mosq_i_t_daily, data_daily, neg_its = interp.weekly_to_daily_spline(
        i_t_weekly, denoised.values, rel_smooth_fac=0.01, spline_degree=5
    )

    daily_t_labels = mosq_day_0 + pd.TimedeltaIndex(mosq_i_t_daily, unit="d")



    neg_its: list  # List of i_t_daily values that had negative interpolations. For debugging.

    # TG-Mosquito data pairing
    # ----------------------------------------------------------------------------------------------------------------

    # # -()- [MIAMI-] Currently, the interpolated mosquito timerange entirely contains the Tg time range. So I'll do this:
    # mosq_it_series = pd.Series(data_daily, index=mosq_i_t_daily)
    # mosq_paired = mosq_it_series.loc[i_t_daily]  # Crop using the Tg timerange
    # tg_paired = tg.df  # Tg in this case is simply the whole data frame

    # -()- [KEYWEST-LOSANGELES-MARICOPA] This one assumes that **tg contains the entire mosquito (series) range**
    mosq_paired = pd.Series(data_daily, index=daily_t_labels)
    tg_paired = tg.df.loc[mosq_paired.index]

    # # ------------- DEBUG WATCH # ------------------------#   # 3 # -
    #   # 3 # -
    # #   # 3 # -
    # # # #            # # # #  # 3 # -

    # Or, the entire season by date
    fig, ax = plt.subplots()
    ax.plot(series, label="Raw")
    ax.plot(series.index, denoised, label="Denoised")
    # ax.plot(daily_t_labels, 7 * data_daily, label="Interpolated * 7")
    # ax.plot(daily_t_labels, data_daily, label="Interpolated")
    ax.plot(tg_paired.index, mosq_paired.values, label="Interpolated (export)")

    # Plot negative interpolation values
    min_, max_ = series.min(), series.max()
    for i_t_neg in neg_its:
        ax.plot(2*[daily_t_labels[i_t_neg]], [min_, max_], "k--", alpha=0.3)# Plot days that had negative interpolations

    # plt.ion()  # Interactive mode on, so it won't block
    ax.legend()
    ax.grid()
    plt.show()
    # plt.pause(0.001)
    # plt.ioff()
    # return

    # \\ =-- -- -\  \\ \\ --\ \ \
    # \ 'ldld[ [] ]\ \ \\\ \ \ \ \\\\\\\\\\\\

    # Export preprocessed mosquito data
    # ---------------------------------
    if export_preprocessed is not None:
        # The time labels of this series agree with those from the MCMC R(t) file
        os.makedirs(os.path.dirname(export_preprocessed), exist_ok=True)
        export_sr = pd.Series(mosq_paired.values, index=tg_paired.index, name="mosq_preproc")
        print(export_sr)
        export_sr.to_csv(export_preprocessed, index_label="date")

    # Exit if not meant to calculate
    if not do_calculate_mcmc:
        return

    # MCMC R(t) estimation
    # -------------------------------------------------------------------------------------------
    mcmc_out_path = mcmc_out_pref + species.replace(" ", "") + "_rt.out"

    if rerun_mcmc:  # Takes time, y'know...
        print("Running MCMC")
        xt0 = time.time()
        rtm = mcmcrt.run_mcmc_rt(mosq_paired.values, tg_paired, out_prefix=mcmc_out_pref, species=species)
        xtf = time.time()
        print(f"Done. Time = {xtf-xt0:0.2}s")

    else:  # Read from precalculated file
        print(f"Loading MCMC from precalculated file: {mcmc_out_path}")
        rtm = mcmcrt.McmcRtEnsemble(np.loadtxt(mcmc_out_path), rolling_width=7, quantile_q=0.025)

    # Index the MCMC results by date
    # rtm.series.rename(lambda i: tg.series.index[i], inplace=True)
    rtm.df = rtm.df.transpose().set_index(tg_paired.index).transpose()

    # Get relevant metrics and Separate by years
    # -------------------------------------------------------------------
    # DF with relevant metrics
    rt_plotables = pd.DataFrame({
        "median": rtm.get_median(),
        "lo_quant": rtm.get_quantiles()[0],
        "hi_quant": rtm.get_quantiles()[1],
        "counts": mosq_paired.values,
    }, index=tg_paired.index)

    # Split by year
    df_list = list()
    for year in years:
        df_list.append(
            # rtm.series.loc[pd.Timestamp(f"{year}-01-01"):pd.Timestamp(f"{year + 1}-01-01")]
            rt_plotables.loc[pd.Timestamp(f"{year}-01-01"):pd.Timestamp(f"{year + 1}-01-01")]
        )
    year_0 = years[0]

    # Finally plots!
    # ----------------------------------------------------------------------------------
    print(df_list)
    print(f"Effective day_0 = {mosq_paired.index[0]}")

    fig, axes = plt.subplots(nrows=2, figsize=(8., 6.4))

    for i_year, year in enumerate(years):
        displaced_dates = df_list[i_year].index - pd.Timedelta(i_year * 365, "d")
        series = df_list[i_year]

        # --- R(t) axes
        ax = axes[0]

        line = ax.plot(displaced_dates, series["median"], label=f"{year}")[0]
        ax.fill_between(displaced_dates, series["lo_quant"], series["hi_quant"], alpha=0.20, color=line.get_color())

        # --- Cases axes
        ax = axes[1]
        line = ax.plot(displaced_dates, series["counts"], label=f"{year}")

    # Figure setup
    xticks = pd.to_datetime([f"{years[0]}-{m}-01" for m in range(1, 13)])
    axes[0].set_xticks(xticks)
    axes[0].axes.xaxis.set_ticklabels([])
    # vis.rotate_ax_labels(axes[0])

    axes[0].set_ylim(0.30, 3.25)
    axes[0].set_ylabel("R(t)")
    axes[0].legend()
    axes[0].grid(True)

    #
    axes[1].set_ylabel("Mosquito counts")
    axes[1].set_xticks(xticks)
    vis.rotate_ax_labels(axes[1])
    axes[1].grid(True)
    axes[1].text(0.6, 0.7, f"Cutoff = {filter_cutoff:0.2f}", transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(f"tmp_figs/rt_years/cutoff{filter_cutoff:0.2f}_".replace(".", "p") + ".pdf")
    plt.show()


if __name__ == "__main__":
    main()
