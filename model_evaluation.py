# Script for model evaluation figure comparing the mooring to the model

import xarray as xr
import xmitgcm
import numpy as np
import mooring_time_series_analyses as mtsa
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_model_evaluation_hov(fp, dt):
    """For creating Hovmoeller diagrams of the plume as well as
    streamfunctions showing the accross-lead view."""

    # Define the start of the plume
    # We'll plot 24 hours before this and 36 hours after
    start = datetime(2021, 9, 13, 12)

    # Open the output
    pref = ['S', 'T', 'U']
    ds = xmitgcm.open_mdsdataset(fp, prefix=pref, delta_t=dt, ref_date=start)
    ds['Z'] = ds['Z'].astype('<f4')  # Endianness
    
    # Streamfunction
    ds['transp'] = (ds['U']*ds['drF']*ds['dyG']).sum("YC")  # Check if vars are correct
    sf = ds['transp'].cumsum(dim='Z')

    # Slice of time (i.e., select a "mooring" location within the domain)
    ds = ds.isel(XC=297, YC=16)

    # Add 24 h mean of pre-plume for visualisation purposes
    def add_24h_mean_prof(ds, start):
        init_conds = ds.isel(time=0)
        tds = [timedelta(hours=i) for i in np.linspace(0, 22, 12)]
        times = [start-timedelta(hours=24) + td for td in tds]
        dss = []
        for time in times:
            dss.append(init_conds.assign_coords(time=time))
        ds = xr.concat(dss + [ds], dim='time')
        return ds
    ds = add_24h_mean_prof(ds, start)

    # Open the mooring data for comparison
    moords = mtsa.open_mooring_data()
    moords = mtsa.correct_mooring_salinities(moords)
    moords = mtsa.append_gsw_vars(moords)
    elapsed_dts = slice(start-timedelta(hours=24), start+timedelta(hours=36))
    moords = moords.sel(time=elapsed_dts)
    moords = moords.rename({'depth': 'Z'})

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters (since mpl uses inches)
    layout = [['a1', 'a2', 'a3'], ['a4', 'a5', 'a6']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3 = ad['a1'], ad['a2'], ad['a3']
    ax4, ax5, ax6 = ad['a4'], ad['a5'], ad['a6']
    fig.set_figwidth(18*cm)
    fig.set_figheight(10*cm)

    # Add some gsw variables to the model data
    ds['p'] = gsw.p_from_z(ds['Z'], -69.0005)
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['t'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['p'])

    # Colours and other misc
    T_min, T_max = -1.8, -0.2
    S_min, S_max = 34.61, 34.79
    T_cmap = mpl.colormaps['Blues_r']
    S_cmap = mpl.colormaps['Oranges']
    T_norm = plt.Normalize(T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)

    # First we split 4 of the panels (this code comes from the google AI)
    def split_panels(ax):
        spec = ax.get_subplotspec()
        ax.set_visible(False)
        gs = spec.subgridspec(1, 3, width_ratios=[24, 2, 36], wspace=0)
        ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[2])]
        ax[0].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].tick_params(axis='y', which='both', left=False, labelleft=False)
        return ax
    ax1 = split_panels(ax1)
    ax2 = split_panels(ax2)
    ax4 = split_panels(ax4)
    ax5 = split_panels(ax5)

    def plot_hov(ds, var, ax, norm, cm, start):
        d = [-50, -125, -220]
        for a in ax:
            p = ds[var].sel(Z=d, method='nearest').plot.contourf(
                x='time', y='Z', ax=a, levels=20, norm=norm, cmap=cm,
                add_colorbar=False)
            a.set_ylim(-220, -50)
        ax[0].set_xlim(start-timedelta(hours=24), start)
        ax[1].set_xlim(start, start+timedelta(hours=36))
        return p

    model_t = plot_hov(ds, 't', ax4, T_norm, T_cmap, start)
    model_s = plot_hov(ds, 'S', ax5, S_norm, S_cmap, start)
    moord_t = plot_hov(moords, 'T', ax1, T_norm, T_cmap, start)
    moord_s = plot_hov(moords, 'SA', ax2, S_norm, S_cmap, start)

    print("The streamfunction stuff could all be wrong!")
    # Note to self: Check that you're using the correct variabiles and coords,
    # also check that the units make sense etc.
    sf.sel(time=start+timedelta(hours=4)).plot.contourf(ax=ax3, levels=20)
    sf.sel(time=start+timedelta(hours=24)).plot.contourf(ax=ax6, levels=20)
    #            x='time', y='Z', ax=a, levels=20, norm=norm, cmap=cm,
    #            add_colorbar=False)


    axes = [ax1[0], ax1[1], ax2[0], ax2[1], ax3,
            ax4[0], ax4[1], ax5[0], ax5[1], ax6]
    for ax in axes:
        ax.set_ylabel("")
        ax.set_yticks([-50, -125, -220])
        ax.set_yticklabels([])
        ax.set_title("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=8)
        ax.grid()
    for ax in [ax1, ax2, ax4, ax5]:
        ax[0].set_xticks(
            [datetime(2021, 9, 12, 12), datetime(2021, 9, 13, 12)])
        ax[1].set_xticks(
            [datetime(2021, 9, 13, 12), datetime(2021, 9, 14, 12)])
        ax[0].set_xticklabels(["Sep 12\n12:00", ""])
        ax[1].set_xticklabels(["Sep 13\n12:00", "Sep 14\n12:00"])

    ax1[0].set_yticklabels(['50 m', '125 m', '220 m'], fontsize=8)
    ax4[0].set_yticklabels(['50 m', '125 m', '220 m'], fontsize=8)

    def add_colourbar(ax, p, label, ticks):
        # See: https://matplotlib.org/stable/gallery/axes_grid1/
        # demo_colorbar_with_inset_locator.html
        axins = inset_axes(
            ax,
            width="220%",
            height="5%",
            loc="upper center",
            bbox_to_anchor=(0.8, 0., 1, 1.35),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(p, orientation='horizontal', cax=axins)
        cbar.ax.set_xticks(ticks)
        cbar.solids.set_edgecolor("face")
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.text(
            0.5,
            1.5,
            label,
            transform=cbar.ax.transAxes,
            fontsize=8,
            ha='center',
        )
    add_colourbar(
        ax1[0], moord_t, "Potential temperature, "+r"$\theta$ "+"($℃$)",
        [-1.8, -1.4, -1, -0.6, -0.2])
    add_colourbar(
        ax2[0], moord_s, "Absolute salinity (g kg$^{-1}$)",
        [34.61, 34.67, 34.73, 34.79])

    # Finally some annotations
    def add_annots(ax, label, c):
        ax.text(
            0.02,
            0.02,
            label,
            transform=ax.transAxes,
            fontsize=8,
            ha='left',
            color=c,
        )
    add_annots(ax1[0], '24 h\n'+r'$\bf{before}$'+'\nthe plume', c='k')
    add_annots(ax1[1], r'$\bf{Observed}$'+'\nplume', c='k')
    add_annots(ax2[0], '24 h\n'+r'$\bf{before}$'+'\nthe plume', c='w')
    add_annots(ax2[1], r'$\bf{Observed}$'+'\nplume', c='w')
    add_annots(ax4[0], 'Model\n'+r'$\bf{initial}$'+'\nconds.', c='k')
    add_annots(ax4[1], r'$\bf{Modelled}$'+'\nplume', c='k')
    add_annots(ax5[0], 'Model\n'+r'$\bf{initial}$'+'\nconds.', c='w')
    add_annots(ax5[1], r'$\bf{Modelled}$'+'\nplume', c='w')

    plt.subplots_adjust(hspace=0.4, wspace=0.16, top=0.8)
    plt.savefig("figure_model_eval.svg")#, transparent=True)
    plt.clf()


if __name__ == "__main__":
    fp = "../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_099"
    plot_model_evaluation_hov(fp, dt=4)
