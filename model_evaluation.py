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
    fig.set_figwidth(25*cm)
    fig.set_figheight(11*cm)

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
        p = ds[var].sel(Z=[-50, -126, -222], method='nearest').plot.contourf(
            x='time', y='Z', ax=ax, levels=20, norm=norm, cmap=cm,
            add_colorbar=False)
        ax.set_ylim(-220, -50)
        ax.set_xlim(datetime(2021, 9, 12, 12), datetime(2021, 9, 15))
        return p

    def plot_moor_hov(moords, var, ax, norm, cm):
        p = moords[var].sel(depth=[-50, -125, -220]).plot.contourf(
            x='time', y='depth', ax=ax, levels=20, norm=norm, cmap=cm,
            add_colorbar=False)
        return p

    model_t = plot_model_hov(ds, 't', ax4, T_norm, T_cmap)
    model_s = plot_model_hov(ds, 'S', ax5, S_norm, S_cmap)
    model_d = plot_model_hov(ds, 'sig0', ax6, sig0_norm, sig0_cmap)

    moord_t = plot_moor_hov(moords, 'T', ax1, T_norm, T_cmap)
    moord_s = plot_moor_hov(moords, 'SA', ax2, S_norm, S_cmap)
    moords['sig0'] = gsw.sigma0(moords['SA'], moords['CT'])
    moord_d = plot_moor_hov(moords, 'sig0', ax3, sig0_norm, sig0_cmap)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_ylabel("")
        ax.set_yticks([-50, -125, -220])
        ax.set_yticklabels([])
        ax.set_xticks([datetime(2021, 9, 12, 12), datetime(2021, 9, 13, 12),
                       datetime(2021, 9, 14, 12)])
        ax.set_xticklabels(["Sep 12\n12:00", "Sep 13\n12:00", "Sep 14\n12:00"])
        ax.set_title("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=9)
        ax.grid()

    ax2.set_title("Observed", fontsize=12)
    ax5.set_title("Modelled", fontsize=12)
    ax1.set_yticklabels(['50 m', '125 m', '220 m'])
    ax4.set_yticklabels(['50 m', '125 m', '220 m'])

    def add_colourbar(ax, p, label):
        cbar = plt.colorbar(p, orientation='vertical', ax=ax)
        cbar.ax.tick_params(labelsize=9)
        bb = dict(fc='white', ec='none', boxstyle='round,pad=0.1', alpha=0.7)
        ax.text(0.025, 0.025, label, transform=ax.transAxes, fontsize=12,
                bbox=bb)

    add_colourbar(ax1, moord_t,
                  "Pot. temp., "+r"$\theta$ "+"($℃$)")
    add_colourbar(ax2, moord_s,
                  "Abs. salinity ($g$ $kg^{-1}$)")
    add_colourbar(ax3, moord_d,
                  "Pot. dens., "+r"$\sigma_0$ "+"($kg$ $m^{-3}$)")
    add_colourbar(ax4, model_t,
                  "Pot. temp., "+r"$\theta$ "+"($℃$)")
    add_colourbar(ax5, model_s,
                  "Abs. salinity ($g$ $kg^{-1}$)")
    add_colourbar(ax6, model_d,
                  "Pot. dens., "+r"$\sigma_0$ "+"($kg$ $m^{-3}$)")

    plt.subplots_adjust(hspace=0.7, wspace=0.16)
    plt.savefig("figure_model_eval.svg", transparent=True)
    plt.clf()


def plot_model_evaluation_anomaly_profiles(ds, dx, dt):

    # Open the mooring data for comparison
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    moords = moords.sel(
        time=slice(datetime(2021, 9, 12, 12), datetime(2021, 9, 15, 0)))

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters
    layout = [['a1', 'a2'], ['a3', 'a4']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = ad['a1'], ad['a2'], ad['a3'], ad['a4']
    fig.set_figwidth(9*cm)
    fig.set_figheight(11*cm)

    mt1 = moords['T'].sel(
        time=slice(datetime(2021, 9, 12, 12), datetime(2021, 9, 13, 12)),
        depth=[-50, -125, -220]).mean('time').values
    mt2 = moords['T'].sel(
        time=slice(datetime(2021, 9, 13, 12), datetime(2021, 9, 15)),
        depth=[-50, -125, -220]).mean('time').values
    ms1 = moords['SA'].sel(
        time=slice(datetime(2021, 9, 12, 12), datetime(2021, 9, 13, 12)),
        depth=[-50, -125, -220]).mean('time').values
    ms2 = moords['SA'].sel(
        time=slice(datetime(2021, 9, 13, 12), datetime(2021, 9, 15)),
        depth=[-50, -125, -220]).mean('time').values
    dst1 = ds['T'].sel(
        time=slice(datetime(2021, 9, 12, 12),
                   datetime(2021, 9, 13, 12))).mean('time').values
    dst2 = ds['T'].sel(
        time=slice(datetime(2021, 9, 13, 12),
                   datetime(2021, 9, 15))).mean('time').values
    dss1 = ds['S'].sel(
        time=slice(datetime(2021, 9, 12, 12),
                   datetime(2021, 9, 13, 12))).mean('time').values
    dss2 = ds['S'].sel(
        time=slice(datetime(2021, 9, 13, 12),
                   datetime(2021, 9, 15))).mean('time').values

    ax1.plot(mt2-mt1, moords['depth'].sel(depth=[-50, -125, -220]).values)
    ax2.plot(ms2-ms1, moords['depth'].sel(depth=[-50, -125, -220]).values)
    ax3.plot(dst2-dst1, ds['Z'].values)
    ax4.plot(dss2-dss1, ds['Z'].values)
    plt.savefig("test.png")


def plot_model_evaluation_profiles(ds):

    # Uncomment if you want to look at HC by slice
    # print(ds["HC'*drF"].sum("Z").values)
    # for n, i in enumerate(ds['Z'].values):
    #     print(ds["HC'*drF"].isel(time=-1).sel(Z=i).values, n)
    # print(ds["HC'*drF"].isel(time=-1, Z=slice(0, 47)).sum().values)
    # print(ds["HC'*drF"].isel(time=-1, Z=slice(47, 71)).sum().values)
    # print(ds["HC'*drF"].isel(time=-1, Z=slice(71, 76)).sum().values)
    # print(ds["HC'*drF"].isel(time=-1, Z=slice(76, -1)).sum().values)

    # Open the mooring data for comparison
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    start_slice = slice(datetime(2021, 9, 12, 12), datetime(2021, 9, 13, 12))
    end_slice = slice(datetime(2021, 9, 14, 12), datetime(2021, 9, 15, 0))
    moords_start = moords.sel(time=start_slice).mean('time')
    moords_end = moords.sel(time=end_slice).mean('time')

    fig, [ax1, ax2, ax2b, ax3] = plt.subplots(ncols=4)
    cm = 1/2.54  # Inches to centimeters
    fig.set_figwidth(16*cm)
    fig.set_figheight(8*cm)

    # Colours
    awi = [55/256, 167/256, 222/256]
    lmu = [34/256, 137/256, 66/256]

    ds['T'].isel(time=0).plot(y='Z', ax=ax1, c=awi, label="Model start")
    ds['T'].isel(time=-1).plot(y='Z', ax=ax1, c=lmu, label='Model end')
    ax1.scatter(moords_start['pt'].values, moords['depth'], c=awi,
                label='Mooring start\n(initial conditions)')
    ax1.scatter(moords_end['pt'].values, moords['depth'], c=lmu,
                label='Mooring end\n(target end)')
    ax1.legend(ncols=4, fontsize=9, loc=3)

    ds['S'].isel(time=0).plot(y='Z', ax=ax2, c=awi)
    ds['S'].isel(time=-1).plot(y='Z', ax=ax2, c=lmu)
    ax2.scatter(moords_start['SA'].values, moords['depth'], c=awi)
    ax2.scatter(moords_end['SA'].values, moords['depth'], c=lmu)

    da = (-1)*(ds['HC'].isel(time=-1) - ds['HC'].isel(time=0))*ds['drF']
    a = 594*4*33*4
    print("a is not right unless you have 594 x points!!!")
    ax2b.plot(np.cumsum(da.values[::-1])/(a*36*3600), da['Z'].values[::-1],
              c='k')

    ds["HC'"].isel(time=-1).plot(y='Z', ax=ax3, c='k')

    ax1.set_title("Pot.\ntemp.,\n"+r"$\theta$ "+"($℃$)", fontsize=12)
    ax2.set_title("Abs.\nsalinity\n($g$ $kg^{-1}$)", fontsize=12)
    ax2b.set_title("Mean vertical\nheat flux ($W$ $m^{-2}$)")
    ax3.set_title("Heat anomaly,\n$HC_{end}-HC_{start}$\n($TJ$)", fontsize=12)

    for ax in [ax1, ax2, ax2b, ax3]:
        ax.tick_params(axis='both', labelsize=9)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([0, -50, -125, -220, -396])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_ylim(-396, 0)

    ax1.set_yticklabels(['0 m', '50 m', '125 m', '220 m', '396 m'])

    plt.subplots_adjust(wspace=0.09, top=0.8)
    plt.savefig("figure_model_eval_profiles.svg", transparent=False)


def run_plot(fp, dt, dx):
    """Run the plotting function."""

    def calc_hc_depth_levels(ds):
        ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
        ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
        ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
        ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
        ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
        ds['HC'] = ds['rho']*ds['cp']*(ds['t_exact']-(-2))*ds['rA']
        ds['HC'] = ds['HC'].sum(['XC', 'YC'])
        ds["HC'"] = ds['HC'] - ds['HC'].isel(time=0)
        ds["HC'"] = ds["HC'"]/1e12
        ds["HC'*drF"] = ds["HC'"]*ds['drF']
        return ds  # Units of HC and HC' are heat/m of depth

    start = datetime(2021, 9, 13, 12)
    pref = ['S', 'T']
    ds = xmitgcm.open_mdsdataset(fp, prefix=pref, delta_t=dt, ref_date=start)
    ds['Z'] = ds['Z'].astype('<f4')  # Endianness
    ds = calc_hc_depth_levels(ds)
    ds = ds.isel(XC=297, YC=16)  # Select point to plot

    def add_24h_mean_prof(ds):
        init_conds = ds.isel(time=0)
        tds = [timedelta(hours=i) for i in np.linspace(0, 22, 12)]
        times = [datetime(2021, 9, 12, 12) + td for td in tds]
        dss = []
        for time in times:
            dss.append(init_conds.assign_coords(time=time))
        ds = xr.concat(dss + [ds], dim='time')
        return ds

    # plot_model_evaluation_hov(add_24h_mean_prof(ds), dx, dt)
    # plot_model_evaluation_anomaly_profiles(add_24h_mean_prof(ds), dx, dt)
    plot_model_evaluation_profiles(ds)


if __name__ == "__main__":
    # fp = "/albedo/work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_094"
    fp = "../MITgcm/so_plumes/mrb_112"
    # fp = "../MITgcm/so_plumes/mrb_101"
    run_plot(fp, dt=4, dx=4)
