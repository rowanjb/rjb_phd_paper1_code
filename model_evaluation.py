# Script for model evaluation figure comparing the mooring to the model

import xmitgcm
import mooring_time_series_analyses as mtsa
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime


def plot_model_evaluation_hov(ds, dx, dt):

    # Open the mooring data for comparison
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    moords = moords.sel(
        time=slice(datetime(2021, 9, 13, 12), datetime(2021, 9, 15, 0)))

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters
    layout = [['a1', 'a2', 'a3'], ['a4', 'a5', 'a6']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3 = ad['a1'], ad['a2'], ad['a3']
    ax4, ax5, ax6 = ad['a4'], ad['a5'], ad['a6']
    fig.set_figwidth(35*cm)
    fig.set_figheight(15*cm)

    ds['p'] = gsw.p_from_z(ds['Z'], -69.0005)
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['t'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['p'])
    ds['sig0'] = gsw.sigma0(ds['S'], ds['CT'])

    # Colours and other misc
    T_min, T_max = -1.8, -0.2
    S_min, S_max = 34.61, 34.79
    sigma0_min, sigma0_max = 27.735, 27.787
    T_cmap = mpl.colormaps['Blues_r']
    S_cmap = mpl.colormaps['Oranges']
    sigma0_cmap = mpl.colormaps['Purples']
    T_title = 'In situ temp.'
    dTdt_title = "$dT/dt$"
    S_title = 'Abs. salinity'
    dSdt_title = "$dS/dt$"
    sigma0_title = 'Pot. dens., $\sigma_0$'
    T_units = '$\degree C$'
    dTdt_units = '$℃$ $d^{-1}$'
    S_units = '$g$ $kg^{-1}$'
    dSdt_units = '$g$ $kg^{-1}$ $d^{-1}$'
    sigma0_units = '$kg$ $m^{-3}$'
    T_norm = plt.Normalize(T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    sigma0_norm = plt.Normalize(sigma0_min, sigma0_max)

    ds = ds.isel(XC=291, YC=16)

    ds['t'].sel(Z=[-50, -126, -222]).plot.contourf(
        x='time', y='Z', ax=ax1, levels=20, norm=T_norm, cmap=T_cmap)
    ax1.set_ylim(-220, -50)
    moords['T'].sel(depth=[-50, -125, -220]).plot.contourf(
        x='time', y='depth', ax=ax4, levels=20, norm=T_norm, cmap=T_cmap)

    ds['S'].sel(Z=[-50, -126, -222]).plot.contourf(
        x='time', y='Z', ax=ax2, levels=20, norm=S_norm, cmap=S_cmap)
    ax2.set_ylim(-220, -50)
    moords['SA'].sel(depth=[-50, -125, -220]).plot.contourf(
        x='time', y='depth', ax=ax5, levels=20, norm=S_norm, cmap=S_cmap)

    ds['sig0'].sel(Z=[-50, -126, -222]).plot.contourf(
        x='time', y='Z', ax=ax3, levels=20, norm=sigma0_norm, cmap=sigma0_cmap)
    ax3.set_ylim(-220, -50)
    moords['sigma0'] = gsw.sigma0(moords['SA'], moords['CT'])
    moords['sigma0'].sel(depth=[-50, -125, -220]).plot.contourf(
        x='time', y='depth', ax=ax6, levels=20, norm=sigma0_norm, cmap=sigma0_cmap)

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

    ds['t'].isel(XC=297, YC=16).sel(Z=-50).plot(x='time', ax=ax1, c='r', ls=':')
    moords['T'].sel(depth=-50).plot(x='time', ax=ax1, c='b', ls=':')

    ds['t'].isel(XC=297, YC=16).sel(Z=-126).plot(x='time', ax=ax1, c='r', ls='--')
    moords['T'].sel(depth=-125).plot(x='time', ax=ax1, c='b', ls='--')

    ds['t'].isel(XC=297, YC=16).sel(Z=-222).plot(x='time', ax=ax1, c='r')
    moords['T'].sel(depth=-220).plot(x='time', ax=ax1, c='b')

    ds['S'].isel(XC=297, YC=16).sel(Z=-50).plot(x='time', ax=ax2, c='r', ls=':')
    moords['SA'].sel(depth=-50).plot(x='time', ax=ax2, c='b', ls=':')

    ds['S'].isel(XC=297, YC=16).sel(Z=-126).plot(x='time', ax=ax2, c='r', ls='--')
    moords['SA'].sel(depth=-125).plot(x='time', ax=ax2, c='b', ls='--')

    ds['S'].isel(XC=297, YC=16).sel(Z=-222).plot(x='time', ax=ax2, c='r')
    moords['SA'].sel(depth=-220).plot(x='time', ax=ax2, c='b')

    plt.savefig('figure_model_eval_time_series.png', dpi=600)
    plt.clf()


def run_plot(fp, dt, dx):
    """Run the plotting function."""

    start = datetime(2021, 9, 13, 12)
    pref = ['S', 'T']
    ds = xmitgcm.open_mdsdataset(fp, prefix=pref, delta_t=dt, ref_date=start)
    ds['Z'] = ds['Z'].astype('<f4')  # Endianness
    plot_model_evaluation_hov(ds, dx, dt)
    # plot_model_evaluation_series(ds, dx, dt)


if __name__ == "__main__":
    #fp = "../MITgcm/so_plumes/mrb_090"
    fp = '/albedo/work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_089'
    run_plot(fp, dt=4, dx=4)
