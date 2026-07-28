# Function plotting the data relating to the Weddell Sea mooring.
# THIS IS PROBABLY DELETABLE 

import analysis_mooring_time_series as amts
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle


def plot_mooring(ds):

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

    # Init the plot
    cm = 1/2.54  # Inches to centimeters
    layout = [
        ['ax1', 'ax1', 'ax3', 'ax3', 'ax3'],
        ['ax2', 'ax2', 'ax4', 'ax4', 'ax4']]
    fig, axd = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = axd['ax1'], axd['ax2'], axd['ax3'], axd['ax4']
    fig.set_figwidth(18*cm)
    fig.set_figheight(12*cm)

    # Colours and other misc formatting stuff
    T_min, T_max = -1.8, -0.2
    S_min, S_max = 34.61, 34.79
    T_cmap = mpl.colormaps['Blues_r']
    S_cmap = mpl.colormaps['Oranges']
    a_title = '(a) September 2021'
    b_title = "(b) Full year"
    c_title = '(c) September 2021'
    d_title = "(d) Full year"
    T_units = '℃'
    S_units = 'g kg$^{-1}$'
    T_norm = plt.Normalize(T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    dates = (datetime(2021, 9, 1, 0), datetime(2021, 10, 1, 0))
    dates_yr = (datetime(2021, 4, 1, 0), datetime(2022, 4, 1, 0))
    T_ticks = [-1.8, -1.4, -1.0, -0.6, -0.2]
    T_labels = ['-1.8', '-1.4', '-1.0', '-0.6', '-0.2']
    S_ticks = [34.61, 34.67, 34.73, 34.79]
    S_labels = ['34.61', '34.67', '34.73', '34.79']

    # For plotting the hovmoellers
    def plotter(da, ax, norm, cmap, levels=20, add_colorbar=False):
        p1 = da.plot.contourf(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar, cmap=cmap, zorder=1,
            rasterized=True)
        return p1

    # For adding a specific black line to really show plume behaviours
    def spec_line(da, ax, value, minmax=False):
        da = da.rolling(time=6).mean()
        if not minmax:
            cm = mpl.colormaps['Greys']
        else:
            cm = mpl.colormaps['Set2']
        da.plot.contour(
            'time', 'depth', ax=ax, levels=1, vmin=value, vmax=value,
            add_colorbar=False, cmap=cm, zorder=2,
            linestyles='solid', linewidths=0.5)

    # ax1: temperature
    da = ds['T'].sel(depth=[-50, -125, -220])
    p1a = plotter(da, ax1, T_norm, T_cmap)
    spec_line(da, ax1, T_min, minmax=True)
    spec_line(da, ax1, -1.3)
    spec_line(da, ax1, T_max, minmax=True)
    ax1.set_xlim(dates)

    # ax2: salinity
    da = ds['SA'].sel(depth=[-50, -125, -220])
    p2a = plotter(da, ax2, S_norm, S_cmap)
    spec_line(da, ax2, S_min, minmax=True)
    spec_line(da, ax2, 34.65)
    spec_line(da, ax2, S_max, minmax=True)
    ax2.set_xlim(dates)

    # ax3
    da = ds['T'].sel(depth=[-50, -125, -220])
    p1a = plotter(da, ax3, T_norm, T_cmap)
    spec_line(da, ax3, T_min, minmax=True)
    spec_line(da, ax3, -1.3)
    spec_line(da, ax3, T_max, minmax=True)
    #ax3.set_xlim(dates_yr)

    # ax4
    da = ds['SA'].sel(depth=[-50, -125, -220])
    p2a = plotter(da, ax4, S_norm, S_cmap)
    spec_line(da, ax4, S_min, minmax=True)
    spec_line(da, ax4, 34.65)
    spec_line(da, ax4, S_max, minmax=True)
    #ax4.set_xlim(dates_yr)

    # Hide unnecessary axis labels and add grid and ticks etc etc
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(labelsize=8)
    for ax in [ax1, ax2]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(zorder=3)
        ax.set_ylim(-220, -50)
        ax.set_yticks([-50, -125, -220])
        ax.set_yticklabels(["50 m", "125 m", "220 m"])
        ax.set_xticks([
            datetime(2021, 9, 1), datetime(2021, 9, 7), datetime(2021, 9, 14),
            datetime(2021, 9, 21), datetime(2021, 9, 28)])
        ax.set_xticklabels(['Sep', '7', '14', '21', '28'])
    #for ax in [ax3, ax4]:
    #    ax.set_ylabel('')
    #    ax.set_xlabel('')
    #    ax.grid(zorder=3)
    #    ax.set_xticks([
    #        datetime(2021, 9, 1), datetime(2021, 9, 3), datetime(2021, 9, 5),
    #        datetime(2021, 9, 7), datetime(2021, 9, 9), datetime(2021, 9, 11),
    #        datetime(2021, 9, 13), datetime(2021, 9, 15), datetime(2021, 9, 17),
    #        datetime(2021, 9, 19), datetime(2021, 9, 21)])
    #    ax.set_xticklabels(
    #        ['Sep', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21'])
    #    ax.spines['top'].set_visible(False)
    #    ax.spines['right'].set_visible(False)

    # Colourbars
    def mk_cbar(ax, p, units, xticks, xticklabels):
        cax = ax.inset_axes([0.05, 1.3, 0.9, 0.08], zorder=400)
        c = fig.colorbar(
            p, cax, orientation='horizontal', extend='both',
            format=ticker.FormatStrFormatter('%.2f'))
        c.ax.set_title(units, rotation=0, fontsize=8)
        c.solids.set_edgecolor("face")
        c.ax.set_xticks(xticks)
        c.ax.set_xticklabels(xticklabels)
        c.ax.tick_params(labelsize=8, rotation=0)
    mk_cbar(ax1, p1a, a_title + ' (' + T_units + ')', T_ticks, T_labels)
    mk_cbar(ax2, p2a, c_title + ' (' + S_units + ')', S_ticks, S_labels)

    # Add title annotations
    ax3.set_title(b_title, fontsize=8)
    ax4.set_title(d_title, fontsize=8)
    ax3.set_ylabel(T_units, fontsize=8)
    ax4.set_ylabel(S_units, fontsize=8)

    # Adjust spacing
    plt.subplots_adjust(left=0.08, hspace=1, wspace=2, right=0.95, top=0.84)

    # Saving
    # plt.savefig('figures/figure_mooring_alternate.pdf', transparent=False, dpi=600)
    plt.savefig('figures/figure_mooring_alternate.svg', transparent=False, dpi=600)
    plt.savefig('figures/figure_mooring_alternate.png', dpi=600)


if __name__ == "__main__":
    ds = amts.open_mooring_data()
    ds = amts.correct_mooring_salinities(ds)
    ds = amts.append_gsw_vars(ds)

    # I slice the dates here so that the final svg is smaller
    ds = ds.sel(time=slice(datetime(2021, 8, 1), datetime(2021, 11, 30)))

    plot_mooring(ds)
