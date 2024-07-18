"""Data visualization helper methods"""
import os
import sys
from collections import defaultdict

import matplotlib as mpl
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES
from rtrend_tools.forecast_structs import ForecastPost, MosqData, TgData
from rtrend_tools.utils import map_parallel_or_sequential
from toolbox.plot_tools import make_axes_seq, get_color_cycle_list

# plt.switch_backend("Agg")  # Prevents on Mac: "NSWindow drag regions should only be invalidated on the Main Thread!"


# ----------------------------------------------------------------------------------------------------------------------
# PLOT FUNCTIONS: For post processed data
# ----------------------------------------------------------------------------------------------------------------------

def get_forecast_cmap(base_cmap="turbo", min_alpha=0.05):
    """A colormap suited for plotting forecast as density."""
    basecmap = mpl.cm.get_cmap(base_cmap)
    norm = None  # mpl.colors.Normalize(vmin=0, vmax=0.1)

    # # Create my own: DARK GREEN MONOTONE
    # cdict = {
    #     "red": [
    #         (0.0, 1.0, 1.0),  # Each row here are the (x, y_left, y_right) coordinates of a piecewise linear
    #         (1.0, 0.0, 0.0),  #   function with this color.
    #     ], "green": [
    #         (0.0, 1.0, 1.0),
    #         (1.0, 0.4, 0.4),
    #     ], "blue": [
    #         (0.0, 1.0, 1.0),
    #         (1.0, 0.0, 0.0),
    #     ], "alpha": [
    #         (0.0, 0.0, 0.0),
    #         (min_alpha, 1.0, 1.0),
    #         (1.0, 1.0, 1.0),
    #     ]
    # }
    # cmap = mpl.colors.LinearSegmentedColormap("ForeGreen", cdict)

    # Create myu own: BASED ON EXISTING CMAP
    def f(val):
        return basecmap(val)

    grid = np.linspace(0.0, 1.0, 10)  # Uniform values from 0 to 1

    cdict = {
        "red":   [(x, f(x)[0], f(x)[0]) for x in grid],
        "green": [(x, f(x)[1], f(x)[1]) for x in grid],
        "blue":  [(x, f(x)[2], f(x)[2]) for x in grid],
        "alpha": [
            (0.0, 0.0, 0.0),
            (min_alpha, 0.1, 1.0),
            (1.0, 1.0, 1.0),
        ]
    }
    cmap = mpl.colors.LinearSegmentedColormap("Fore" + base_cmap, cdict)

    return cmap, norm


def plot_forecasts_as_density(ct_fore2d: np.ndarray, ax: mpl.axes.Axes, i_t_pres=0):

    # Preamble
    # --------

    num_samples, num_days_fore = ct_fore2d.shape
    max_ct = ct_fore2d.max()  # Maximum value in th y-axis

    dens_matrix = np.empty((num_days_fore, max_ct + 1), dtype=float)  # Signature | a[i_t][sample]

    # Calculation
    # -----------

    # Density histogram (unsing bincount) for each time step
    # for i_t, day in enumerate(range(i_t_pres, i_t_pres + num_days_fore)):
    for i_t in range(num_days_fore):
        rt_ensemble = ct_fore2d[:, i_t]

        bc = np.bincount(rt_ensemble, minlength=max_ct + 1)
        dens_matrix[i_t, :] = bc / num_samples   # -()- NORMALIZES AT EACH TIME STEP

    # Plot
    # ----
    # ax.imshow(dens_matrix.T, interpolation="bilinear")
    t_array = np.arange(i_t_pres, i_t_pres + num_days_fore)
    y_array = np.arange(0, max_ct + 1)

    cmap, norm = get_forecast_cmap()
    dens = ax.pcolormesh(t_array, y_array, dens_matrix.T, shading="gouraud", cmap=cmap, norm=norm, edgecolor="none")
    # shading: gouraud, auto (or nearest)

    return dens, cmap


# Paired quantiles (EXCLUDES THE MEDIAN q = 0.5)
QUANTILE_TUPLES = [(CDC_QUANTILES_SEQ[i], CDC_QUANTILES_SEQ[-i - 1]) for i in range(int(NUM_QUANTILES / 2))]


def plot_forecasts_median(ct_fore2d: np.ndarray, ax: mpl.axes.Axes, i_t_pres=0,
                          color="green", alpha=1.0):

    num_samples, num_days_fore = ct_fore2d.shape
    t_array = np.arange(i_t_pres, i_t_pres + num_days_fore)

    return ax.plot(t_array, np.median(ct_fore2d, axis=0), ls="--", color=color, alpha=alpha)


def plot_forecasts_as_quantile_layers(ct_fore2d: np.ndarray, ax: mpl.axes.Axes, i_t_pres=0,
                                      color="green", base_alpha=0.08, quant_tuples=None):
    # Preamble
    # --------
    num_samples, num_days_fore = ct_fore2d.shape

    if quant_tuples is None:
        quant_tuples = QUANTILE_TUPLES

    layers = list()
    t_array = np.arange(i_t_pres, i_t_pres + num_days_fore)

    # Calculation and plot
    # --------------------

    # Quantiles
    for i_layer, q_pair in enumerate(quant_tuples):
        low_quant = np.quantile(ct_fore2d, q_pair[0], axis=0)
        hig_quant = np.quantile(ct_fore2d, q_pair[1], axis=0)

        layers.append(ax.fill_between(t_array, low_quant, hig_quant, color=color, alpha=base_alpha, lw=1))

    return layers


# ----------------------------------------------------------------------------------------------------------------------
# PLOT FUNCTIONS: For post processed data
# ----------------------------------------------------------------------------------------------------------------------


# Generic plot function for a whole single page
def plot_page_generic(chunk_seq, plot_func, page_width=9., ncols=3, ax_height=3., plot_kwargs=None):
    """
    Signature of the callable:
        plot_func(ax, item_from_chunk, i_ax, **plot_kwargs)
    """
    n_plots = len(chunk_seq)
    plot_kwargs = dict() if plot_kwargs is None else plot_kwargs

    fig, axes = make_axes_seq(n_plots, ncols, total_width=page_width, ax_height=ax_height)
    plot_outs = list()

    for i_ax, (ax, item) in enumerate(zip(axes, chunk_seq)):
        plot_outs.append(plot_func(ax, item, i_ax, **plot_kwargs))

    return fig, axes, plot_outs


def plot_precalc_quantiles_as_layers(
        ax, quant_lines: np.ndarray, x_array, alpha=0.1, color="green", linewidth=1):
    """
    Plots a sequence of lines as transparent filled layers.
    Lines are paired from the edges of quant_lines to its center ((0, -1), (1, -2), etc...).
    If the number of lines is odd, the central one is not included. ((n-1)/2)


    """
    num_layers = quant_lines.shape[0] // 2
    layers = list()

    for i_q in range(num_layers):
        layers.append(ax.fill_between(x_array, quant_lines[i_q], quant_lines[-(i_q + 1)],
                                      alpha=alpha, color=color, linewidth=linewidth))

    return layers


# ----------------------------------------------------------------------------------------------------------------------
# HIGH LEVEL PLOT SCRIPTS
# ----------------------------------------------------------------------------------------------------------------------

def plot_ct_past_and_fore(ax, fore_time_labels: pd.DatetimeIndex, weekly_quantiles, factual_ct: pd.Series,
                          plot_name="", i_ax=None, synth_name=None,
                          num_quantiles=None, ct_color="C0", insert_point=None, bkg_color=None,
                          quantile_alpha=0.08, quantile_linewidth=1.):
    """Plot all data and configure ax for C(t) data of a single state."""

    if not isinstance(fore_time_labels, pd.DatetimeIndex):  # Tries to convert into a pandas Index if not yet
        fore_time_labels = pd.DatetimeIndex(fore_time_labels)

    if num_quantiles is None:
        num_quantiles = weekly_quantiles.shape[0]

    # Optionally inserts the "today" point for better visualization of the forecast
    fore_x_data = fore_time_labels
    fore_y_data2d = weekly_quantiles
    if insert_point is not None:
        fore_x_data = fore_x_data.insert(0, insert_point[0])  # Includes present date
        fore_y_data2d = np.insert(fore_y_data2d, 0, insert_point[1], axis=1)

    # Plot C(t) forecast quantiles and median.
    q_layers = plot_precalc_quantiles_as_layers(
        ax, fore_y_data2d, fore_x_data, alpha=quantile_alpha, linewidth=quantile_linewidth)
    if num_quantiles % 2 == 1:
        median = ax.plot(fore_x_data, fore_y_data2d[num_quantiles // 2], "g--s", ms=2.5)
    else:
        median = None

    # Factual C(t) time series (past and, if available, forecast)
    ax.plot(factual_ct, "--o", color=ct_color, ms=2.5)

    # Write state and synth name
    # ax.text(0.05, 0.9, species, transform=ax.transAxes)
    text = plot_name if i_ax is None else f"{i_ax + 1}) {plot_name}"
    ax.text(0.05, 0.9, text, transform=ax.transAxes)

    # Write name of the synth method
    if synth_name is not None:
        ax.text(0.05, 0.8, synth_name, transform=ax.transAxes)

    ax.set_xticks(factual_ct.index)
    rotate_ax_labels(ax)
    
    if bkg_color is not None:
        ax.set_facecolor(bkg_color)

    return q_layers, median


def make_plot_tables_sweepdates(post_list: list[ForecastPost], mosq: MosqData, tg: TgData, dates_ens: pd.DatetimeIndex,
                                params, args, ncols=3, nrows=3, write_synth_names=True):
    """
    Produces multiple tables of plots for the sweep_forecast_dates.py.
    """
    if not params.misc["do_plots"]:
        return

    print("\n___________________\nNOW PLOTTING\n")
    os.makedirs("tmp_figs", exist_ok=True)

    # PLOTS PREAMBLE
    # -------------------------------------------------------------------
    # Filter dates without forecast (ignored)
    content_zip = [(i_date, post) for i_date, post in enumerate(post_list) if post is not None]
    num_filt_items = len(content_zip)

    # Prepare the subdivision of plots into pages
    axes_per_page = ncols * nrows  # Maximum number of plots per page
    npages_states = (num_filt_items - 1) // axes_per_page + 1  # Number of pages for the states plots
    naxes_last = (num_filt_items - 1) % axes_per_page + 1  # Number of plots on the last page

    # Group items by pages
    content_chunks = [content_zip[axes_per_page * i: axes_per_page * (i + 1)] for i in range(npages_states - 1)]
    content_chunks.append(content_zip[-naxes_last:])

    # PLOTS - FORECAST C(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_forecast_ct(ax: plt.Axes, item, i_ax):
        i_post, post = item  # Unpack content tuple
        post: ForecastPost
        day_pres: pd.Timestamp = post.day_pres

        # Contents
        factual_ct: pd.Series = mosq.df.loc[post.day_0:post.day_fore][post.species]
        last_val = post.data_weekly.iloc[-1]  # species_series.iloc[-1]  # From actual or preproc data
        ct_color = "C1"  # if day_pres < mosq.data_time_labels[-1] else "C0"

        # Plot function
        plot_ct_past_and_fore(ax, post.fore_time_labels, post.weekly_quantiles, factual_ct, day_pres.date(), i_post,
                              post.synth_name if write_synth_names else None, post.num_quantiles, ct_color,
                              (day_pres, last_val), bkg_color="#E9F7DF", quantile_alpha=0.028)

        # Plot scores
        if post.score_wis is not None:
            txt = "rWIS = " + ", ".join((f"{s:0.2f}" for s in post.score_rwis))
            ax.text(0.05, 0.74, txt, transform=ax.transAxes, fontdict={"size": 6})

        # Extra x-axis setup
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6, interval=4))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=6, interval=1))

        # Plot also denoised data
        ax.plot(post.denoised_weekly)

        print(f"  [{day_pres.date()}] ({i_ax+1} of {num_filt_items})  |", end="")
        sys.stdout.flush()

    def task(chunk):
        return plot_page_generic(chunk, plot_table_forecast_ct, ncols=ncols)

    print("Plotting forecasts...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=params.misc["ncpus"])  # Not faster?
    print()

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/sweepdate_ct.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)
            # fig.savefig("tmp_figs/ct_states.png")  # EXTRA: also saves as png

    # PLOTS - PAST AND FUTURE R(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_rt(ax: plt.Axes, item, i_ax):
        i_post, post = item  # Unpack content tuple
        past_t = post.past_daily_tlabels[-params.mcmc["roi_len_days"]:]

        # Estimated R(t) from past
        ax.plot(past_t, post.rt_past_median, color="C0")
        ax.plot(past_t, post.rt_past_mean, color="C0", ls="--")
        ax.fill_between(past_t, post.rt_past_loquant, post.rt_past_hiquant, color="C0", alpha=0.25)

        # Synthesised R(t)
        ax.plot(post.fore_daily_tlabels, post.rt_fore_median, color="blue")
        ax.fill_between(post.fore_daily_tlabels, post.rt_fore_loquant, post.rt_fore_hiquant, color="blue",
                        alpha=0.15)

        # 1-line
        ax.plot([past_t[0], post.fore_daily_tlabels[-1]], [1, 1], "k--", alpha=0.5)

        # Plot annotations
        ax.text(0.05, 0.9, f"{i_ax + 1}) {post.day_pres.date()}", transform=ax.transAxes)
        if write_synth_names:
            ax.text(0.05, 0.8, post.synth_name, transform=ax.transAxes)

        # Axes configuration
        rotate_ax_labels(ax)
        ax.set_ylim(0.0, 3.1)
        ax.set_facecolor("#E9F7DF")
        print(f"  [{post.day_pres.date()}] ({i_ax + 1} of {num_filt_items})  |", end="")
        sys.stdout.flush()

    def task(chunk):
        return plot_page_generic(chunk, plot_table_rt, ncols=ncols)

    print("\nPlotting R(t)...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=params.misc["ncpus"])  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/sweepdate_rt.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)

    print()
    print("Plots done!")


# ----------------------------------------------------------------------------------------------------------------------
# MOSQUITO PAPER STYLE
# ----------------------------------------------------------------------------------------------------------------------
mosq_color_cycle_01 = [
    '#63CC78', '#A978CC', '#5AADCC', '#CC8550', '#A8CC5B',
]

# Adapted from https://colormagic.app/ with keyword: "scientific graph"
# Colorblind-safe checked via Adobe Color
mosq_color_cycle_02 = [
    "#3cafa3",  # Medium pool-green
    "#f6d55a",  # CAASO yellow
    "#1f616f",  # Dark green
    "#ed563b",  # Red-orange-ish
    "#593B85",  # Dark violet
]

# A color palette for schematic figures. Used in Illustrator.
scheme_colors_01 = [
    "#1a3542",  # Dark green
    "#3cafa3",  # Bright green
    "#ed563b",  # Orangish red
    "#1f616f",  # Medium green
    "#f6d55a",  # CAASO yellow
    "#333333",  # Almost black
    "#b2b2b2",  # Gray-ish
]

aux_green_color = "#4E965A"  # Darker shade to contrast with #63CC78

mosquito_paper_rc = {
    # --- Fonts
    #     "font.family": "DejaVu Sans",
    "font.family": "Open Sans",
    # font.style:   "normal",
    # font.variant: "normal",
    "font.weight": "light",
    "font.stretch": "condensed",
    "font.size": 12.0,

    # --- Figure Props
    "figure.figsize": (4., 4.),

    # --- Colors
    "axes.prop_cycle": cycler(color=mosq_color_cycle_01),

    # --- Unwanted linewidths
    "patch.linewidth": 0,
    "lines.markeredgewidth": 0,
    "lines.markeredgecolor": "none",

    # Misc
    "legend.fontsize": "small",
}


# Other color palettes (theme)
# -----------------------------------------------------
location_color_cycle = [
    "#593B85",  # Dark violet
    "#3cafa3",  # Medium pool-green
    "#f6d55a",  # CAASO yellow
    "#ed563b",  # Red-orange-ish
]

# # -()- Colorful scheme for horizons
# weeks_ahead_color_cycle = [
#     "#6b71ed",  # Blue?
#     "#1aab42",  # Definitely green
#     "#ff6161",  # A not so pale red
#     "#f6da74",  # Cute yellow
# ]

# -()- Another colorful scheme for horizons
weeks_ahead_color_cycle = [
    "#E09358",  # Orangeish?
    "#BA58E0",  # Rose
    "#58D1E0",  # Cyan
    "#B8E058",  # Lime
]

# # -()- Monochrome scheme for horizons
# weeks_ahead_color_cycle = [
#     "#148032",  # Green
#     "#1aab42",  # Definitely green
#     "#76cd8e",  # Not so green
#     "#a3ddb3",  # Tired green
# ]

activity_color_cycle = [
    "#42BD4A",  # Bright green
    "#EBD203",  # Yellow
    "#FF603F",  # Red
]


# --- Define a dictionary keyed by feature name.
feat_colors_dict = defaultdict(
    lambda: mosq_color_cycle_01,  # Default color cycle
    weeks_ahead=weeks_ahead_color_cycle,
    location=location_color_cycle,
    activity_index=activity_color_cycle,
)


# ----------------------------------------------------------------------------------------------------------------------
# AUX METHODS
# ----------------------------------------------------------------------------------------------------------------------


def rotate_ax_labels(ax, angle=60, xy="x", which="major"):
    """This function could be included in my plot_tools."""
    labels = ax.get_xticklabels(which=which) if xy == "x" else ax.get_yticklabels(which=which)
    for label in labels:
        label.set(rotation=angle, horizontalalignment='right')
