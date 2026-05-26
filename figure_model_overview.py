# For making an overview figure of the model setup
# Requires some post-processing in inkscape

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib as mpl
import xmitgcm
import gsw
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
import cmocean


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
        ['a1', 'a1', 'a1', 'a1', 'a1',  '.', 'a3', 'a3', 'a4', 'a4', 'a5', 'a5'],
        ['a1', 'a1', 'a1', 'a1', 'a1',  '.', 'a3', 'a3', 'a4', 'a4', 'a5', 'a5'],
        ['a1', 'a1', 'a1', 'a1', 'a1',  '.', 'a3', 'a3', 'a4', 'a4', 'a5', 'a5'],
        ['a2', 'a2', 'a2', 'a2', 'a2',  '.', 'a3', 'a3', 'a4', 'a4', 'a5', 'a5'],
        ['a2', 'a2', 'a2', 'a2', 'a2',  '.', 'a3', 'a3', 'a4', 'a4', 'a5', 'a5'],
        ['a2', 'a2', 'a2', 'a2', 'a2',  '.', 'a3', 'a3', 'a4', 'a4', 'a5', 'a5'],
        [ '.',  '.',  '.',  '.',  '.',  '.',  '.',  '.',  '.',  '.',  '.',  '.'],
        ['aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa',  '.',  '.',  '.',  '.'],
        ['aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa', 'aa',  '.',  '.', 'a7', 'a7'],
        ['a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'ab',  '.', 'a7', 'a7'],
        ['a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'ab',  '.', 'a7', 'a7'],
        ['a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'ab',  '.', 'a7', 'a7'],
        ['a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'ab',  '.', 'a7', 'a7'],
        ['a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'ab',  '.', 'a7', 'a7'],
        ['a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'a6', 'ab',  '.',  '.',  '.']]
    fig, ad = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = ad['a1'], ad['a2'], ad['a3'], ad['a4']
    ax5, ax6, ax7 = ad['a5'], ad['a6'], ad['a7']
    axa, axb = ad['aa'], ad['ab']
    fig.set_figwidth(18*cm)
    fig.set_figheight(14*cm)

    # Open the basic file
    fp = "../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_121"
    start = datetime(2021, 9, 13, 12)
    pref = ['S', 'T', 'U']
    ds = xmitgcm.open_mdsdataset(fp, prefix=pref, delta_t=4, ref_date=start)
    ds['Z'] = ds['Z'].astype('<f4')  # Endianness

    # Plot the flux panels
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
        ax.set_xlabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()
    ax1.set_xticklabels([])
    ax2.set_xticklabels([
        "Sep 13\n12:00",
        "Sep 14\n12:00",
        "Sep 15\n12:00",
        "Sep 16\n12:00",
    ])
    ax1.set_ylim(-0.28, 0.1)
    ax2.set_ylim(-20, 280)

    # Plot the initial conditions panels
    init = ds.isel(time=0, YC=17, XC=297)
    init['CT'] = gsw.CT_from_pt(init['S'], init['T'])
    init['sigma0'] = gsw.sigma0(init['S'], init['CT'])
    obs = init.sel(Z=[-51, -124, -219])
    init['T'].plot(ax=ax3, y='Z', c='k')
    ax3.scatter(x=obs['T'], y=obs['Z'], c='k')
    init['S'].plot(ax=ax4, y='Z', c='k')
    ax4.scatter(x=obs['S'], y=obs['Z'], c='k')
    ax5.scatter(x=obs['sigma0'], y=obs['Z'], c='k')
    init['sigma0'].plot(ax=ax5, y='Z', c='k')
    for ax in [ax3, ax4, ax5]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([0, -50, -125, -220, -396])
        ax.set_yticklabels(['0 m', '50 m', '125 m', '220 m', '396 m'])
        ax.grid()
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])
    ax3.set_xlim([-2.1, 1])
    ax4.set_xlim([34.58, 34.99])
    ax5.set_xlim([27.74, 27.85])

    # Plot the vertical resolution
    drf = ds['drF']
    ax7.plot(drf, np.arange(0, len(drf)), c='k')
    ax7.invert_yaxis()
    ax7.spines['right'].set_visible(False)
    ax7.spines['top'].set_visible(False)
    ax7.set_ylabel("Vertical level", fontsize=8)
    ax7.grid()

    # Plot the "primary" slice of the field and the colourbar
    da = ds['S'].isel(time=3)
    cmap = plt.cm.colors.ListedColormap(cmocean.cm.haline(np.linspace(0.01, 1, 256)))
    vmin, vmax = 34.625, 34.685
    C = da.isel(YC=0).plot.contourf(levels=19,
        ax=ax6, cmap=cmap, vmin=vmin, vmax=vmax, extend='max', add_colorbar=False)
    axins = inset_axes(
        ax6, width="35%", height="4%", loc='lower center',
        bbox_to_anchor=(0, 0.125, 1, 1), bbox_transform=ax6.transAxes)
    cbar = fig.colorbar(C, cax=axins, orientation='horizontal', extend=False)
    axins.set_xticks([34.625, 34.645, 34.665, 34.685])
    axins.set_title(r'(f) Abs. salinity after 6 hours (g kg$^{-1}$)', size=8)
    axins.tick_params(labelsize=8)
    cbar.solids.set_edgecolor("face")
    ax6.set_yticks([0, -50, -125, -220])
    ax6.set_yticklabels(['0 m', '50 m', '125 m', '220 m'])
    ax6.set_ylim([-220, 0])
    ax6.set_xlim([594*2-300, 594*2+300])
    ax6.set_xticks([594*2-300, 594*2, 594*2+300])
    ax6.set_xticklabels(
        [str(594*2-300)+' m', str(594*2)+' m', str(594*2+300)+' m'])
    ax6.set_xlabel('')
    ax6.set_ylabel('')

    # bbox in parent axis coordinates for the colourbar
    box = FancyBboxPatch(
        (0.30, 0.18), 0.40, 0,
        boxstyle="Round,pad=0.07,rounding_size=0.035", fc="white",
        ec="none", alpha=0.6, transform=ax6.transAxes, zorder=2)
    box.set_mutation_aspect(8 / 4)
    ax6.add_patch(box)

    # Additional surfaces (modify later in inkscape!)
    da.isel(Z=0).plot.contourf(levels=19,
        ax=axa, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    da.isel(XC=372).plot.contourf(levels=19,
        ax=axb, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    for ax in [axa, axb]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('')
    axa.set_yticks([0, 66, 132])
    axa.set_yticklabels(['0 m', '66 m', '132 m'])
    axa.set_xlim([594*2-300, 594*2+300])
    axb.set_ylim([-220, 0])

    # Adding titles etc
    bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6)
    ax1.set_title("Buoyancy forcing through sea ice lead", fontsize=8)
    ax1.text(
        0.5, 0.86, "(a) Salt flux",
        va='center', ha='center',
        rotation='horizontal',
        transform=ax1.transAxes,
        fontsize=8, bbox=bb)
    ax1.set_ylabel(r"g m$^{-2}$ s$^{-1}$", fontsize=8)
    ax2.text(
        0.5, 0.86, "(b) Heat flux",
        va='center', ha='center',
        rotation='horizontal',
        transform=ax2.transAxes,
        fontsize=8, bbox=bb)
    ax2.set_ylabel(r"W m$^{-2}$", fontsize=8)
    ax3.set_title("", fontsize=8)
    ax3.text(
        0.5, 0.9, "(c) Pot.\ntemp.",
        va='center', ha='center',
        rotation='horizontal',
        transform=ax3.transAxes,
        fontsize=8, bbox=bb)
    ax3.set_xlabel(r"℃", fontsize=8)
    ax4.set_title("Initial conditions", fontsize=8)
    ax4.text(
        0.5, 0.9, "(d) Abs.\nsalinity",
        va='center', ha='center',
        rotation='horizontal',
        transform=ax4.transAxes,
        fontsize=8, bbox=bb)
    ax4.set_xlabel(r"g kg$^{-1}$", fontsize=8)
    ax5.set_title("", fontsize=8)
    ax5.text(
        0.5, 0.9, "(e) Pot.\ndensity",
        va='center', ha='center',
        rotation='horizontal',
        transform=ax5.transAxes,
        fontsize=8, bbox=bb)
    ax5.set_xlabel(r"kg$^{-1}$ m$^{-3}$", fontsize=8)
    ax7.set_title("", fontsize=8)
    ax7.text(
        0.5, 0.9, "(g) Vertical\nresolution",
        va='center', ha='center',
        rotation='horizontal',
        transform=ax7.transAxes,
        fontsize=8, bbox=bb)
    ax6.set_title("", fontsize=8)

    # Some basic formatting for all axes
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, axa, axb]
    for ax in axes:
        ax.tick_params(axis='both', labelsize=8)

    plt.subplots_adjust(
        right=0.97, left=0.1, hspace=0.5, wspace=0.4, top=0.95, bottom=0.085)
    plt.savefig("figure_model_overview.png", dpi=500)
    plt.savefig("figure_model_overview.svg")
    plt.savefig("figure_model_overview.pdf")


if __name__ == "__main__":
    plot_model_overview()
