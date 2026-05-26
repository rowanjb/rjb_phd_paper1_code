# Script for model evaluation figure comparing the mooring to the model

import xarray as xr
import xmitgcm
import numpy as np
import analysis_mooring_time_series as mtsa
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_model_evaluation_hov(fp, dt):
    """For creating Hovmoeller diagrams of the plume as well as
    streamfunctions showing the accross-lead view."""

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
    layout = [
        ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', '.', 'a3', 'a3', 'a3'],
        ['a4', 'a4', 'a4', 'a5', 'a5', 'a5', '.', 'a6', 'a6', 'a6']]
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
    sf_min, sf_max = -100, 100
    T_cmap = mpl.colormaps['Blues_r']
    S_cmap = mpl.colormaps['Oranges']
    sf_cmap = mpl.colormaps['bwr']
    T_norm = plt.Normalize(T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    sf_norm = plt.Normalize(sf_min, sf_max)

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

    # Plotting the hovmoellers
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

    # Plotting the streamfunctions
    # (Check that the units make sense?)
    t_top_hrs, t_bot_hrs = 4, 24
    sf_top = sf.sel(time=start+timedelta(hours=t_top_hrs)).plot.contourf(
        ax=ax3, levels=20, norm=sf_norm, cmap=sf_cmap, add_colorbar=False)
    sf_bot = sf.sel(time=start+timedelta(hours=t_bot_hrs)).plot.contourf(
        ax=ax6, levels=20, norm=sf_norm, cmap=sf_cmap, add_colorbar=False)

    # Handling formatting etc.
    axes = [ax1[0], ax1[1], ax2[0], ax2[1],
            ax4[0], ax4[1], ax5[0], ax5[1],]
    for ax in axes:
        ax.set_ylabel("")
        ax.set_yticks([-50, -125, -220])
        ax.set_yticklabels([])
        ax.set_title("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=8)
        ax.grid()
    for ax in [ax1, ax2, ax4, ax5]:  # Have to adjust manually is start changes
        ax[0].set_xticks(
            [datetime(2021, 9, 12, 12), datetime(2021, 9, 13, 12)])
        ax[1].set_xticks(
            [datetime(2021, 9, 13, 12), datetime(2021, 9, 14, 12)])
        ax[0].set_xticklabels(["Sep 12\n12:00", ""])
        ax[1].set_xticklabels(["Sep 13\n12:00", "Sep 14\n12:00"])
    for ax in [ax3, ax6]:
        dx = sf['XG'].isel(XG=-1).to_numpy()/2
        ax.set_xticks([0, dx, dx*2])
        ax.set_xticklabels(
            [str(int((-1)*dx))+' m', str(0)+' m', str(int(dx))+' m'])
        ax.set_yticks([0, -50, -125, -220, sf['Z'].isel(Z=-1).data])
        ax.set_yticklabels(['0 m', '50 m', '125 m', '220 m',
                            str(abs(sf['Z'].isel(Z=-1).data))+' m'])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title("")
        ax.tick_params(axis='both', labelsize=8)
        ax.grid()
    ax1[0].set_yticklabels(['50 m', '125 m', '220 m'], fontsize=8)
    ax4[0].set_yticklabels(['50 m', '125 m', '220 m'], fontsize=8)
    ax6.set_xlabel("Across-lead distance", fontsize=8)

    # Colourbars for the split axes
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
    
    # Colourbar for the streamfunctions
    axins = inset_axes(
        ax3, width="90%", height="5%", loc="upper center", borderpad=0,
        bbox_to_anchor=(0, 0., 1, 1.35), bbox_transform=ax3.transAxes)
    cbar = fig.colorbar(sf_top, orientation='horizontal', cax=axins)
    cbar.ax.set_xticks([-100, -50, 0, 50, 100])
    cbar.solids.set_edgecolor("face")
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.text(
        0.5, 1.5, "Across-lead\nstreamfunction (m$^3$ s$^{-1}$)",
        transform=cbar.ax.transAxes, fontsize=8, ha='center') 

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
    t_top_annot = ('$t_{model}=$'+str(t_top_hrs)+' h,\n' +
                   str((start+timedelta(hours=t_top_hrs)).strftime(
                       "%b %d %H:%M"))+'')
    add_annots(ax3, t_top_annot, c='k')
    t_bot_annot = ('$t_{model}=$'+str(t_bot_hrs)+' h,\n' +
                   str((start+timedelta(hours=t_bot_hrs)).strftime(
                       "%b %d %H:%M"))+'')
    add_annots(ax6, t_bot_annot, c='k')

    # Add panel lettering
    def add_letter(ax, x, y, letter):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=8, va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='none',
                          boxstyle='circle,pad=0.05',
                          alpha=0.2))
    add_letter(ax1[0], 0.28, 0.95, '(a)')
    add_letter(ax2[0], 0.28, 0.95, '(b)')
    add_letter(ax3,    0.11,  0.95, '(e)')
    add_letter(ax4[0], 0.28, 0.95, '(c)')
    add_letter(ax5[0], 0.28, 0.95, '(d)')
    add_letter(ax6,    0.11,  0.95, '(f)')

    plt.subplots_adjust(hspace=0.4, wspace=0.5, top=0.8, left=0.1, right=0.93)
    plt.savefig("figure_model_eval.pdf")
    plt.savefig("figure_model_eval.svg", transparent=True)
    plt.savefig("figure_model_eval.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    fp = "../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_121"
    #fp = "../MITgcm/so_plumes/mrb_121"
    plot_model_evaluation_hov(fp, dt=4)
