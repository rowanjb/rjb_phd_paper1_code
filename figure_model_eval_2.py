# Script for model evaluation figure comparing the mooring to the model
# Here we want to quantify how accurate the model is, and justify using
# one primary simulation in the later analysis of the paper

import xarray as xr
import f90nml
import xmitgcm
import numpy as np
import pandas as pd
import analysis_mooring_time_series as mtsa
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Get delta_t
def get_delta_t(fp):
    nml = f90nml.read(fp+"/data")
    return nml["PARM03"]["deltaT"]


# Opening the model dataset
def open_dataset(fp):
    # Note here we're assuming a mooring in the centre of the lead
    dt = get_delta_t(fp)
    ds = xmitgcm.open_mdsdataset(fp, prefix=['S', 'T'], delta_t=dt)
    ds = ds.isel(XC=int(len(ds['XC'])/2), YC=int(len(ds['YC'])/2))
    ds['Z'] = ds['Z'].astype('<f4')
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['sigma0'] = gsw.sigma0(ds['S'], ds['CT'])
    ds = ds.interp(Z=[-50, -125, -220])
    return ds


def plot_model_evaluation_time_series():
    """For creating time series of the model-observation error."""

    # IPCC-adjacent formatting according to ChatGPT
    mpl.rcParams.update({
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'axes.grid': False,
        'grid.linewidth': 0.5,
        'grid.color': '0.85',
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'legend.frameon': False,
        'mathtext.default': 'regular',
        'svg.fonttype': 'none',
    })

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters (since mpl uses inches)
    layout = [
        ['a1', 'a2'],
        ['a1', 'a2'],
        ['a3', 'a4'],
        ['a3', 'a4'],
        ['a5', 'a6'],
        ['a5', 'a6'],
        [ '.',  '.'],
        ['a7', 'a7'],
        ['a7', 'a7'],
        ['a7', 'a7']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = ad['a1'], ad['a2'], ad['a3'], ad['a4']
    ax5, ax6, ax7 = ad['a5'], ad['a6'], ad['a7']
    fig.set_figwidth(18*cm)
    fig.set_figheight(15*cm)

    # The observation time series
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    start = datetime(2021, 9, 13, 12)
    elapsed_dts = slice(start, start+timedelta(hours=72))
    moords = moords.sel(time=elapsed_dts)
    moords = moords.rename({'depth': 'Z'})
    moords = moords.sel(Z=[-50, -125, -220])

    # Plot the model time series
    # The dataframe syntax was helped by ChatGPT
    simulations = [  # You'll eventually need to update this with the updated list
        'mrb_121_1', 'mrb_120', 'mrb_122', 'mrb_123', 'mrb_124', 'mrb_125',
        'mrb_126', 'mrb_129', 'mrb_130', 'mrb_131', 'mrb_132', 'mrb_133',
        'mrb_134', 'mrb_135', 'mrb_136', 'mrb_137', 'mrb_138', 'mrb_139',
        'mrb_140', 'mrb_141', 'mrb_142', 'mrb_143', 'mrb_144']
    S50, S125, S220, T50, T125, T220 = {}, {}, {}, {}, {}, {}
    for simulation in simulations:
        fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'
        ds = open_dataset(fp + simulation)
        ds['time'] = moords['time'].isel(time=slice(0, len(ds['time'])))  # Sometimes the model is short
        ds['S'] = ds['S'] - moords['SA'].isel(time=slice(0, len(ds['time'])))
        ds['CT'] = ds['CT'] - moords['CT'].isel(time=slice(0, len(ds['time'])))
        time = ds['time'].to_numpy()
        S50[simulation] = (time, np.abs(ds['S'].sel(Z=-50).to_numpy()))
        S125[simulation] = (time, np.abs(ds['S'].sel(Z=-125).to_numpy()))
        S220[simulation] = (time, np.abs(ds['S'].sel(Z=-220).to_numpy()))
        T50[simulation] = (time, np.abs(ds['CT'].sel(Z=-50).to_numpy()))
        T125[simulation] = (time, np.abs(ds['CT'].sel(Z=-125).to_numpy()))
        T220[simulation] = (time, np.abs(ds['CT'].sel(Z=-220).to_numpy()))

    # Convert dictionaries to DataFrames
    S50_df = pd.DataFrame()
    S125_df = pd.DataFrame()
    S220_df = pd.DataFrame()
    T50_df = pd.DataFrame()
    T125_df = pd.DataFrame()
    T220_df = pd.DataFrame()
    for sim, (t, values) in S50.items():
        S50_df[sim] = pd.Series(values, index=t)
    for sim, (t, values) in S125.items():
        S125_df[sim] = pd.Series(values, index=t)
    for sim, (t, values) in S220.items():
        S220_df[sim] = pd.Series(values, index=t)
    for sim, (t, values) in T50.items():
        T50_df[sim] = pd.Series(values, index=t)
    for sim, (t, values) in T125.items():
        T125_df[sim] = pd.Series(values, index=t)
    for sim, (t, values) in T220.items():
        T220_df[sim] = pd.Series(values, index=t)

    # Compute the RMSE of each time series at each depth from each simulation
    S50_rmse = np.sqrt((S50_df**2).mean(axis=0))
    S125_rmse = np.sqrt((S125_df**2).mean(axis=0))
    S220_rmse = np.sqrt((S220_df**2).mean(axis=0))
    T50_rmse = np.sqrt((T50_df**2).mean(axis=0))
    T125_rmse = np.sqrt((T125_df**2).mean(axis=0))
    T220_rmse = np.sqrt((T220_df**2).mean(axis=0))

    # Combine the RMSE metrics into one dataframe for each of processing
    metrics = pd.DataFrame({
        'S50': S50_rmse,
        'S125': S125_rmse,
        'S220': S220_rmse,
        'T50': T50_rmse,
        'T125': T125_rmse,
        'T220': T220_rmse,
    })

    # Normalize each metric and score the simulations
    metrics_norm = metrics / metrics.mean(axis=0)
    score = metrics_norm.mean(axis=1).sort_values()

    # Plotting the data
    colour = (112/256, 160/256, 205/256)
    S50_df.plot(ax=ax1, legend=False, color=colour)
    S125_df.plot(ax=ax3, legend=False, color=colour)
    S220_df.plot(ax=ax5, legend=False, color=colour)
    T50_df.plot(ax=ax2, legend=False, color=colour)
    T125_df.plot(ax=ax4, legend=False, color=colour)
    T220_df.plot(ax=ax6, legend=False, color=colour)
    score.plot.bar(ax=ax7, color=colour)

    # Highlighting a specific run
    runid = 'mrb_121_1'
    S50_df[runid].plot(ax=ax1, legend=False, color='k')
    S125_df[runid].plot(ax=ax3, legend=False, color='k')
    S220_df[runid].plot(ax=ax5, legend=False, color='k')
    T50_df[runid].plot(ax=ax2, legend=False, color='k')
    T125_df[runid].plot(ax=ax4, legend=False, color='k')
    T220_df[runid].plot(ax=ax6, legend=False, color='k')
    run_idx = score.index.get_loc(runid)
    ax7.patches[run_idx].set_color('k')

    # Adding numbers according to ChatGPT
    for bar in ax7.patches:
        height = bar.get_height()
        ax7.text(
            bar.get_x() + bar.get_width()/2,
            height/2,                 # middle of the bar
            f'{height:.3f}',
            fontsize=8,
            ha='center',
            va='center',
            rotation=90,
            color='white'              # optional: visible on the bar
        )

    # Handling formatting etc.
    xticks = [start + timedelta(hours=12*int(i)) for i in np.arange(7)]
    xticklabels = ['' if i % 2 else t.strftime('%b %d\n%H:%M')
                   for i, t in enumerate(xticks)]
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=8)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xticks(xticks)
        ax.grid()
        ax.set_xticklabels(xticklabels)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([])
    ax7.set_axisbelow(True)
    ax7.grid(axis='y')
    ax7.tick_params(axis='x', rotation=30)
    for label in ax7.get_xticklabels():
        label.set_ha('right')

    # Add panel letters and titles
    ax1.set_title("Conservative temperature error (℃)", fontsize=8)
    ax2.set_title("Absolute salinity error (g kg$^{-1}$)", fontsize=8)
    def add_title(ax, title):
        ax.text(
            0.5, 0.9, title, transform=ax.transAxes, fontsize=8, va='top',
            ha='center', bbox=dict(facecolor='white', edgecolor='none',
            boxstyle='round,pad=0.1', alpha=0.5))
    add_title(ax1, '(a) 50 m sensor')
    add_title(ax2, '(b) 50 m sensor')
    add_title(ax3, '(c) 125 m sensor')
    add_title(ax4, '(d) 125 m sensor')
    add_title(ax5, '(e) 220 m sensor')
    add_title(ax6, '(f) 220 m sensor')
    add_title(ax7, '(g) Normalised model-observation mean RMSE')

    plt.suptitle("Model-observation differences", fontsize=8)
    plt.subplots_adjust(
        hspace=0.35, wspace=0.3, top=0.9, left=0.1, right=0.93)
    plt.savefig("figures/figure_model_eval_2.svg", transparent=False)
    plt.savefig("figures/figure_model_eval_2.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    plot_model_evaluation_time_series()
