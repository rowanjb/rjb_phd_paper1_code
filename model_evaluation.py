# Script for model evaluation figure comparing the mooring to the model

import xarray as xr
import xmitgcm
import numpy as np
import mooring_time_series_analyses as mtsa
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta


def plot_model_evaluation_hov(ds, dx, dt):

    # Open the mooring data for comparison
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    moords = moords.sel(
        time=slice(datetime(2021, 9, 12, 12), datetime(2021, 9, 15, 0)))

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters
    layout = [['a1', 'a2', 'a3'], ['a4', 'a5', 'a6']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3 = ad['a1'], ad['a2'], ad['a3']
    ax4, ax5, ax6 = ad['a4'], ad['a5'], ad['a6']
    fig.set_figwidth(35*cm)
    fig.set_figheight(15*cm)

    # Add some gsw variables to the model data
    ds['p'] = gsw.p_from_z(ds['Z'], -69.0005)
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['t'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['p'])
    ds['sig0'] = gsw.sigma0(ds['S'], ds['CT'])

    # Colours and other misc
    T_min, T_max = -1.8, -0.2
    S_min, S_max = 34.61, 34.79
    sig0_min, sig0_max = 27.735, 27.787
    T_cmap = mpl.colormaps['Blues_r']
    S_cmap = mpl.colormaps['Oranges']
    sig0_cmap = mpl.colormaps['Purples']
    T_norm = plt.Normalize(T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    sig0_norm = plt.Normalize(sig0_min, sig0_max)

    def plot_model_hov(ds, var, ax, norm, cm):
        ds[var].sel(Z=[-50, -126, -222]).plot.contourf(
            x='time', y='Z', ax=ax, levels=20, norm=norm, cmap=cm)
        ax.set_ylim(-220, -50)
        ax.set_xlim(datetime(2021, 9, 12, 12), datetime(2021, 9, 15))

    def plot_moor_hov(moords, var, ax, norm, cm):
        moords[var].sel(depth=[-50, -125, -220]).plot.contourf(
            x='time', y='depth', ax=ax, levels=20, norm=norm, cmap=cm)

    plot_model_hov(ds, 't', ax1, T_norm, T_cmap)
    plot_model_hov(ds, 'S', ax2, S_norm, S_cmap)
    plot_model_hov(ds, 'sig0', ax3, sig0_norm, sig0_cmap)

    plot_moor_hov(moords, 'T', ax4, T_norm, T_cmap)
    plot_moor_hov(moords, 'SA', ax5, S_norm, S_cmap)
    moords['sig0'] = gsw.sigma0(moords['SA'], moords['CT'])
    plot_moor_hov(moords, 'sig0', ax6, sig0_norm, sig0_cmap)

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig("old_figs_and_scripts/figure_model_eval.svg")
    plt.clf()


def plot_model_evaluation_series(ds, dx, dt):

    # Open the mooring data for comparison
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    moords = moords.sel(
        time=slice(datetime(2021, 9, 13, 12), datetime(2021, 9, 15, 0)))

    ds['p'] = gsw.p_from_z(ds['Z'], -69.0005)
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['t'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['p'])
    ds['sig0'] = gsw.sigma0(ds['S'], ds['CT'])

    fig, [ax1, ax2] = plt.subplots(ncols=2)

    ds['t'].sel(Z=-50).plot(x='time', ax=ax1, c='r', ls=':')
    moords['T'].sel(depth=-50).plot(x='time', ax=ax1, c='b', ls=':')

    ds['t'].sel(Z=-126).plot(x='time', ax=ax1, c='r', ls='--')
    moords['T'].sel(depth=-125).plot(x='time', ax=ax1, c='b', ls='--')

    ds['t'].sel(Z=-222).plot(x='time', ax=ax1, c='r')
    moords['T'].sel(depth=-220).plot(x='time', ax=ax1, c='b')

    ds['S'].sel(Z=-50).plot(x='time', ax=ax2, c='r', ls=':')
    moords['SA'].sel(depth=-50).plot(x='time', ax=ax2, c='b', ls=':')

    ds['S'].sel(Z=-126).plot(x='time', ax=ax2, c='r', ls='--')
    moords['SA'].sel(depth=-125).plot(x='time', ax=ax2, c='b', ls='--')

    ds['S'].sel(Z=-222).plot(x='time', ax=ax2, c='r')
    moords['SA'].sel(depth=-220).plot(x='time', ax=ax2, c='b')

    plt.savefig('figure_model_eval_time_series.png', dpi=600)
    plt.clf()


def run_plot(fp, dt, dx):
    """Run the plotting function."""

    start = datetime(2021, 9, 13, 12)
    pref = ['S', 'T']
    ds = xmitgcm.open_mdsdataset(fp, prefix=pref, delta_t=dt, ref_date=start)
    ds['Z'] = ds['Z'].astype('<f4')  # Endianness
    ds = ds.isel(XC=297, YC=16)   # Select point to plot

    def add_24h_mean_prof(ds):
        init_conds = ds.isel(time=0)
        tds = [timedelta(hours=i) for i in np.linspace(0, 22, 12)]
        times = [datetime(2021, 9, 12, 12) + td for td in tds]
        dss = []
        for time in times:
            dss.append(init_conds.assign_coords(time=time))
        ds = xr.concat(dss + [ds], dim='time')
        return ds

    plot_model_evaluation_series(ds, dx, dt)
    plot_model_evaluation_hov(add_24h_mean_prof(ds), dx, dt)


if __name__ == "__main__":
    # fp = "/albedo/work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_090"
    fp = "../MITgcm/so_plumes/mrb_092"
    run_plot(fp, dt=4, dx=4)
