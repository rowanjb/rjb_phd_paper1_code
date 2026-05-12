# Function plotting the data relating to the Weddell Sea mooring.

import analysis_mooring_time_series as amts
import xarray as xr
import gsw
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
    T_title = 'In situ temperature'
    dTdt_title = "Temperature rate of change, $dT/dt$"
    S_title = 'Absolute salinity'
    dSdt_title = "Salinity rate of change, $dS/dt$"
    T_units = '℃'
    dTdt_units = '℃ d$^{-1}$'
    S_units = 'g kg$^{-1}$'
    dSdt_units = 'g kg$^{-1}$ d$^{-1}$'
    T_norm = plt.Normalize(T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    dates = (datetime(2021, 9, 1, 0), datetime(2021, 10, 1, 0))
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

    # Filter wrapper for smoothing the high-freq d/dt data
    def savgol_plotter(ds, var, d, ax, deriv=1, window=36, po=3, delta=2/24):
        cmap_query = {
            -50: (0/256, 0/256, 0/256),
            -125: (112/256, 160/256, 205/256),
            -220: (196/256, 121/256, 0/256)}
        ls_query = {-50: '-', -125: "-", -220: '-'}
        da = ds[var].sel(depth=d)
        da_filtered = savgol_filter(
            da.values, window_length=window, polyorder=po, deriv=deriv,
            delta=delta)
        new_da = xr.DataArray(da_filtered, {'time': da['time']})
        new_da = new_da.sel(
            time=slice(datetime(2021, 9, 1), datetime(2021, 9, 21)))
        colour = cmap_query[d]
        ls = ls_query[d]
        p, = new_da.plot(ax=ax, c=colour, lw=1, label=d, ls=ls)
        ax.set_xlim(datetime(2021, 9, 1), datetime(2021, 9, 21))
        return p

    # ax3: temperature ROC
    pT5 = savgol_plotter(ds, 'T', -220, ax3)
    label5 = "220 m"# (T, S)"
    pT3 = savgol_plotter(ds, 'T', -125, ax3)
    label3 = "125 m"# (T, S)"
    pT1 = savgol_plotter(ds, 'T', -50, ax3)
    label1 = "50 m"# (T, S)"

    # ax4: salinity ROC
    pS3 = savgol_plotter(ds, 'SA', -220, ax4)
    pS2 = savgol_plotter(ds, 'SA', -125, ax4)
    pS1 = savgol_plotter(ds, 'SA', -50, ax4)

    # Highlighting the modelled period
    for ax in [ax3, ax4]:
        rect = Rectangle(
            (datetime(2021, 9, 13, 12), -1), timedelta(hours=72),
            2, color="#e5e4e4")
        ax.add_patch(rect)

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
    for ax in [ax3, ax4]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(zorder=3)
        ax.set_xticks([
            datetime(2021, 9, 1), datetime(2021, 9, 3), datetime(2021, 9, 5),
            datetime(2021, 9, 7), datetime(2021, 9, 9), datetime(2021, 9, 11),
            datetime(2021, 9, 13), datetime(2021, 9, 15), datetime(2021, 9, 17),
            datetime(2021, 9, 19), datetime(2021, 9, 21)])
        ax.set_xticklabels(
            ['Sep', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

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
    mk_cbar(ax1, p1a, T_title + ' (' + T_units + ')', T_ticks, T_labels)
    mk_cbar(ax2, p2a, S_title + ' (' + S_units + ')', S_ticks, S_labels)

    # Add title annotations
    ax3.set_title(dTdt_title, fontsize=8)
    ax4.set_title(dSdt_title, fontsize=8)
    ax3.set_ylabel(dTdt_units, fontsize=8)
    ax4.set_ylabel(dSdt_units, fontsize=8)

    # Legend
    handles = [pT1, pT3, pT5]
    labels = [label1, label3, label5]
    ax4.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.3),
        title='Sensor depth', title_fontsize=8,
        frameon=False, fontsize=8, ncol=3)

    # Letter annotations
    labs = {ax1: '(a)', ax2: '(b)', ax3: '(c)', ax4: '(d)'}
    for ax in [ax1, ax2]:
        ax.text(
            -0.15, 1.275, labs[ax], transform=ax.transAxes, fontsize=8,
            va='top', ha='left', zorder=120)
    for ax in [ax3, ax4]:
        ax.text(-0.1, 1.275, labs[ax], transform=ax.transAxes, fontsize=8,
        va='top', ha='left', zorder=120)

    # Adjust spacing
    plt.subplots_adjust(left=0.08, hspace=1, wspace=1.35, right=0.95, top=0.84)

    # Saving
    plt.savefig('figure_mooring.pdf', transparent=False, dpi=600)
    plt.savefig('figure_mooring.svg', transparent=True, dpi=600)
    plt.savefig('figure_mooring.png', dpi=600)


if __name__ == "__main__":
    ds = amts.open_mooring_data()
    ds = amts.correct_mooring_salinities(ds)
    ds = amts.append_gsw_vars(ds)

    # I slice the dates here so that the final svg is smaller
    ds = ds.sel(time=slice(datetime(2021, 8, 1), datetime(2021, 11, 30)))

    plot_mooring(ds)
