import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as feature
from pyhdf.SD import SD, SDC
import cmocean
from matplotlib.patches import Rectangle

import copy
import matplotlib.cm as colourmap
import xarray as xr
import xmitgcm
import numpy as np
import analysis_mooring_time_series as mtsa
import gsw
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projections

import pyvista as pv
from skimage.measure import marching_cubes


def plot_model_overview():
    """Make figure for paper."""
    # Aspects of formatting etc here were helped by ChatGPT and Google AI

    # Model run we're working with
    fp = "../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_121"

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
        [ '.', 'a1', 'a1', 'a1', 'a1',  '.',  '.', 'a2', 'a2', 'a2', 'a2',  '.'],
        [ '.', 'a1', 'a1', 'a1', 'a1',  '.',  '.', 'a2', 'a2', 'a2', 'a2',  '.'],
        [ '.', 'a1', 'a1', 'a1', 'a1',  '.',  '.', 'a2', 'a2', 'a2', 'a2',  '.'],
        [ '.', 'a1', 'a1', 'a1', 'a1',  '.',  '.', 'a2', 'a2', 'a2', 'a2',  '.'],
        [ '.', 'a1', 'a1', 'a1', 'a1',  '.',  '.', 'a2', 'a2', 'a2', 'a2',  '.'],
        ['ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac',  '.',  '.'],
        ['ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac', 'ac',  '.',  '.'],
        ['a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'aa', '.'],
        ['a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'aa', '.'],
        ['a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'aa', '.'],
        ['a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'aa', '.'],
        ['a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'a3', 'aa', '.'],
        [ '.', '.', '.', '.',  '.', '.', '.', '.', '.', '.', '.',  '.'],
        [ '.', 'a4', 'a4', 'a4',  '.', 'a6', 'a6', 'a7', 'a7', 'a8', 'a8',  '.'],
        [ '.', 'a4', 'a4', 'a4',  '.', 'a6', 'a6', 'a7', 'a7', 'a8', 'a8',  '.'],
        [ '.', 'a4', 'a4', 'a4',  '.', 'a6', 'a6', 'a7', 'a7', 'a8', 'a8',  '.'],
        [ '.', 'a4', 'a4', 'a4',  '.', 'a6', 'a6', 'a7', 'a7', 'a8', 'a8',  '.'],
        [ '.', 'a4', 'a4', 'a4',  '.', 'a6', 'a6', 'a7', 'a7', 'a8', 'a8',  '.']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = ad['a1'], ad['a2'], ad['a3'], ad['a4']
    ax6, ax7, ax8 = ad['a6'], ad['a7'], ad['a8']
    axa, axc = ad['aa'], ad['ac']
    fig.set_figwidth(18*cm)
    fig.set_figheight(18*cm)

    # Open the basic file
    fp = "../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_121"
    start = datetime(2021, 9, 13, 12)
    pref = ['S', 'T', 'U']
    ds = xmitgcm.open_mdsdataset(fp, prefix=pref, delta_t=4, ref_date=start)
    ds['Z'] = ds['Z'].astype('<f4')  # Endianness

    # Plot the fluxes
    force_vars = ["Qin", "Sin"]
    force_fps = {
        "Qin": fp+"/bin_forc_Q_72x33x594_100m_lead.bin",
        "Sin": fp+"/bin_forc_SA_72x33x594_100m_lead_037.bin",
    }
    force_axes = {"Sin": ax1, "Qin": ax2}
    for force_var in force_vars:
        force_da = xr.DataArray(
            xmitgcm.utils.read_raw_data(
                force_fps[force_var],
                shape=(72, 33, 594),
                dtype=np.dtype('>f4')
            )[:, 17, 297],
            dims=["time"],
            coords={
                "time": [start + td(hours=int(i)) for i in np.arange(0, 72)]
            }
        )
        ax = force_axes[force_var]
        force_da.plot(ax=ax, c='k')
        ax.set_xticks([
            dt(2021, 9, 13, 12),
            dt(2021, 9, 14, 12),
            dt(2021, 9, 15, 12),
            dt(2021, 9, 16, 12),
        ])
        ax.set_xticklabels([
            "Sep 13\n12:00",
            "Sep 14\n12:00",
            "Sep 15\n12:00",
            "Sep 16\n12:00",
        ])
        ax.set_xlabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()

    # Plot the vertical resolution
    drf = ds['drF'].values
    ax4.plot(drf, np.arange(0, len(drf)), c='k')
    ax4.invert_yaxis()
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_ylabel("Vertical level", fontsize=8)
    ax4.grid()
    ax4.set_yticks([0, 50, 99])
    #ax4.set_xticks([5, 10])
    #ax4.set_xticklabels(['5 m', '10 m'])

    # Plot the initial conditions
    init = ds.isel(time=0, YC=17, XC=297)
    init['T'].plot(ax=ax6, y='Z', c='k')
    init['S'].plot(ax=ax7, y='Z', c='k')
    init['CT'] = gsw.CT_from_pt(init['S'], init['T'])
    init['sigma0'] = gsw.sigma0(init['S'], init['CT'])
    init['sigma0'].plot(ax=ax8, y='Z', c='k')
    for ax in [ax6, ax7, ax8]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([0, -50, -125, -220, -396])
        ax.set_yticklabels(['0 m', '50 m', '125 m', '220 m', '396 m'])
        ax.grid()
    ax7.set_yticklabels([])
    ax8.set_yticklabels([])
    ax6.set_xlim([-2.1, 1])
    ax7.set_xlim([34.5, 34.99])
    ax8.set_xlim([27.74, 27.85])

    # Plot slice of temperature
    T = ds['S'].isel(time=2)
    # copy the colormap
    cmap = copy.copy(colourmap.bone)
    vmin, vmax = 34.61, 34.79
    C = T.isel(YC=0).plot.contourf(levels=19,
        ax=ax3, cmap=cmap, vmin=vmin, vmax=vmax, extend='max', add_colorbar=False)
    axins = inset_axes(
        ax3, width="35%", height="8%", loc='lower center',
        bbox_to_anchor=(0, 0.3, 1, 1), bbox_transform=ax3.transAxes)
    cbar = fig.colorbar(C, cax=axins, orientation='horizontal', extend=False)
    axins.set_xticks([34.61, 34.67, 34.73, 34.79])
    cbar.set_label(r'(c) Abs. salinity after 4 hours (g kg$^{-1}$)', size=8)
    axins.tick_params(labelsize=8)
    cbar.solids.set_edgecolor("face")
    ax3.set_yticks([0, -50, -125, -220, -396])
    ax3.set_yticklabels(['0 m', '50 m', '125 m', '220 m', '396 m'])
    #ax5.set_yticks([0, -50, -125, -220])
    #ax5.set_yticklabels(['0 m', '50 m', '125 m', '220 m'])
    ax3.set_xticks([0, 594*2, 594*4])
    ax3.set_xticklabels(['0 m', str(594*2)+' m', str(594*4)+' m'])
    ax3.set_xlim(0, 594*4)
    #ax5.set_xticks([700, 594*2])
    for ax in [ax3]:#, ax5]:
        ax.set_xlabel('')
        ax.set_ylabel('')

    ## Adding grid lines to help with inkscape annotations
    #axc.hlines(66, 700, 1188, colors='k', ls='--')
    #axc.vlines(700, 4, 66, colors='k', ls='--')
    #axc.vlines(1188, 4, 66, colors='k', ls='--')
    #ax3.hlines(-220, 700, 1188, colors='k', ls='--')
    #ax3.vlines(700, -4, -220, colors='k', ls='--')
    #ax3.vlines(1188, -4, -220, colors='k', ls='--')

    # Additional surfaces (modify later in inkscape!)
    T.isel(XC=-1).plot.contourf(levels=19,
        ax=axa, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    T.isel(Z=0).plot.contourf(levels=19,
        ax=axc, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    for ax in [axa, axc]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('')
    axc.set_yticks([0, 66, 132])
    axc.set_yticklabels(['0 m', '66 m', '132 m'])
    #axd.set_yticks([0, 66])
    #axd.set_yticklabels(['0 m', '66 m'])

    # Adding titles etc
    ax1.set_title("(a) Salt flux, $S_{in}$", fontsize=8)
    ax1.set_ylabel("g m$^{-2}$ s$^{-1}$", fontsize=8)
    ax2.set_title("(b) Heat flux, $Q_{in}$", fontsize=8)
    ax2.set_ylabel("W m$^{-2}$", fontsize=8)
    ax4.set_title("(d) Vertical resolution", fontsize=8)
    ax4.set_xlabel("m", fontsize=8)
    ax6.set_title(r"(e) Pot. temp., $\theta$", fontsize=8)
    ax6.set_xlabel(r"℃", fontsize=8)
    ax7.set_title("(f) Abs. salinity", fontsize=8)
    ax7.set_xlabel(r"g kg$^{-1}$", fontsize=8)
    ax8.set_title(r"(g) Pot. dens., $\sigma_{\theta}$", fontsize=8)
    ax8.set_xlabel(r"kg$^{-1}$ m$^{-3}$", fontsize=8)
    ax3.set_title("(c) Snaptshot of abs. salinity after 4 hours", fontsize=8)
    #ax5.set_title("(e) Close-up of abs. sal.", fontsize=8)

    # Some basic formatting for all axes
    axes = [ax1, ax2, ax3, ax4, ax6, ax7, ax8, axa, axc]
    for ax in axes:
        ax.tick_params(axis='both', labelsize=8)

    # # Add these axes in later
    # for ax in [ax3, ax5]:
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.get_xaxis().set_ticks([])
    #     ax.get_yaxis().set_ticks([])

    plt.subplots_adjust(
        right=0.975, left=0.1, hspace=10, wspace=0.5, top=0.95, bottom=0.085)
    plt.savefig("figure_model_overview.png", dpi=500)
    plt.savefig("figure_model_overview.svg")
    plt.savefig("figure_model_overview.pdf")


if __name__ == "__main__":
    plot_model_overview()
