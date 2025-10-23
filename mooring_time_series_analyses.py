# Functions for opening, processing, and plotting data relating to the
# Weddell Sea mooring. Based on older functions and oop, but slimmed
# down here. Requires a few datasets:
#   (1) Sea ice concentration (see mooring_atm_and_ice.py for details)
#   (2) WOA data
#   (3) Mooring data

import xarray as xr
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as patches
import scipy.io as spio
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cmocean
import matplotlib as mpl


def resample_mooring_data(inmat, jul):
    """Useful in open_mooring_data().
    jul is the original dates, which are irregular (e.g., every 2
    hours +/- 5 mins), hence why we use resample.
    jul is indexed from the start of the year, hence why we have a
    start_date and subsequently add a timedelta object onto this for
    each data point.
    The dates can be tested against the MATLAB datevec() function (since
    all the mooring data is originally from a .mat file), i.e.,
     - first temperature at the 50 m sensor = -1.507 C
     - datevec(first date at the 50 m sensor) = 2021-03-24T12
    Therefore we want our dataset to have -1.507 at 2021-03-24T12."""

    # For getting the variable at dt intervals
    start_date = datetime(2020, 12, 30, 0, 0, 0)  # Indexed from 30/12

    # Looping through each depth level, 0 to 5
    dfs = []
    for sensor_id in range(len(jul)):

        # Put data into a pandas dataframe
        dates = pd.to_datetime(
            [start_date+timedelta(days=i[0]) for i in jul[sensor_id]]
        )

        # If a non-empty list (sometimes data is missing)
        if len(inmat[sensor_id]) > 0:
            var_data = [i[0] for i in inmat[sensor_id]]
        else:
            var_data = [np.nan for i in dates]
        df = pd.DataFrame(
            data={'dates': dates, sensor_id: var_data}
        )
        df = df.set_index('dates')
        df = df[
            (df.index > '2021-03-24 00:00:00') &
            (df.index < '2022-04-07 14:00:00')
        ]

        # Use the very nice pandas resample method
        df = df.resample('h').mean()

        dfs.append(df)

    return pd.concat(dfs, axis=1)


def open_mooring_data():
    """Opens the mooring .mat file(s); returns a dataset"""

    print("Beginning to open the mooring data")

    # Open file path (saved in txt to avoid publishing to GitHub)
    # [0] accesses the first line, and [:-1] removes the newline tag
    with open('../filepaths/mooring_filepath') as f:
        dirpath = f.readlines()[0][:-1]

    # Creating the full filepaths to the .mat files
    # BGC_SBE is the main data, and sal_BGC has corrected salinities
    # for two of the sensors
    print(("Note that Markus is dotting i's and crossing t's re. "
           "checking how the salinities were corrected"))
    filepath_BGC_SBE = dirpath + '/CTD/Mooring/BGC_SBE.mat'
    filepath_sal_BGC = dirpath + '/CTD/Mooring/sal_BGC.mat'
    mat = spio.loadmat(filepath_BGC_SBE)['SBE']  # SBE = Sea Bird
    mat_corr = spio.loadmat(filepath_sal_BGC)

    # Extracting the needed data from the main .mat file
    # Note 'P' is missing data at all but the -50 and -125 sensors
    jul = mat['jul'][0]
    T, S, P = mat['T'][0], mat['S'][0], mat['P'][0]

    # Extracting the corrected salinities and updating the S array
    Sal_449 = mat_corr['Sal_449']
    Sal_2100 = mat_corr['Sal_2100']
    S[4] = Sal_449
    S[2] = Sal_2100

    # Resample to get everything onto an even time "grid" with datetime coords
    T_resampled = resample_mooring_data(T, jul)
    S_resampled = resample_mooring_data(S, jul)
    P_resampled = resample_mooring_data(P, jul)

    # Save it all into an xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            T=(["time", "depth"], T_resampled.to_numpy()),
            S=(["time", "depth"], S_resampled.to_numpy()),
            P=(["time", "depth"], P_resampled.to_numpy()),
        ),
        coords=dict(
            time=T_resampled.index.to_numpy(),
            depth=[-50, -90, -125, -170, -220, -250],
        ),
        attrs=dict(description="Mooring data"),
    )

    # The 50 m sensor only has 2-hourly data, so I'm cutting it down here;
    # basically just removing some nans
    ds = ds.isel(time=slice(0, -1, 2))

    print("Mooring data opened")
    return ds


def correct_mooring_salinities(ds):
    """Unfortunately, the salinities from the lower two sensors seem,
    basically, wrong. I am waiting on word from the observationalists
    about how they calibrated/corrected the salinities originally, but
    until then I will use this function. I suspect they might not have
    had time to calibrate the sensors before launching them.
        Essentially, the lower two salinities from are ~2 PSU too
    fresh, which is unphysical. The corrected salinities from sal_BGC
    have no documentation and also seem somewhat unphysical (i.e.,
    the salt_bgc-corrected salinities lead to a consistent
    potential density inversion between the lower two sensors),
    so in this function I'm basically overwriting the salinities
    by equating the sensors' means with those of a WOA climatology.
    This has the effect of slightly destabilising the water column.
    I'm not happy about this but c'est la vie, because I have few
    alternatives.
        Importantly, this has no bearing on the proof that we have a
    plume, since this is predicated on the ROC of the sensors and not
    their actually values. This does however have a bearing on the model
    initial stratification, and is something that needs to be tested and
    handled carefully.
        Note that one potential source of error, particularly with the
    middle sensor, is that it's not 100% clear if it was at 135 m or
    125 m, another reason to correct it.
        For info on WOA data, see the WORLD OCEAN ATLAS 2023 Product
    Documentation.
    """

    print("Beginning to correct the mooring data")

    # Calculate the mean salinites
    S = ds['S']  # Extract as a dataarray for easy handling
    # S_srfce_mean_mooring = S.sel(depth=-50).mean(dim='time').values
    S_upper_mean_mooring = S.sel(depth=-125).mean(dim='time').values
    S_lower_mean_mooring = S.sel(depth=-220).mean(dim='time').values

    # Open the WOA data
    with open('../filepaths/woa_filepath') as f:
        dirpath = f.readlines()[0][:-1]
    ds_woa = xr.open_dataset(
        dirpath+'/WOA_monthly_'+'s'+'_'+str(2015)+'.nc',
        decode_times=False
    )

    # Times in WOA are saved as arbitrary reals
    ds_woa = ds_woa.rename({'time': 'month'})
    ds_woa['month'] = ds_woa['month'] - 35.5

    # Calculate the yearly average, using weighting procedure from:
    # xarray example "area_weighted_temperature"
    # Note the s_an are practical salinities
    ds_woa['weights'] = (  # Simplistic but works!
        'month', [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    ds_woa_weighted = ds_woa['s_an'].weighted(ds_woa['weights'])
    woa_weighted_mean = ds_woa_weighted.mean('month')

    # Get the WOA salinities at the correct depths
    # S_srfce_mean_woa = woa_weighted_mean.interp(depth=50).values
    S_upper_mean_woa = woa_weighted_mean.interp(depth=125).values
    S_lower_mean_woa = woa_weighted_mean.interp(depth=220).values

    # "Correct" the mooring salinities
    # S_srfce_mean_anomaly = S_srfce_mean_mooring-S_srfce_mean_woa
    S_upper_mean_anomaly = S_upper_mean_mooring-S_upper_mean_woa
    S_lower_mean_anomaly = S_lower_mean_mooring-S_lower_mean_woa
    # S = xr.where(S['depth'] == -50, S.sel(depth=-50)-S_srfce_mean_anomaly, S)
    S = xr.where(S['depth'] == -125, S.sel(depth=-125)-S_upper_mean_anomaly, S)
    S = xr.where(S['depth'] == -220, S.sel(depth=-220)-S_lower_mean_anomaly, S)

    # Reassign the corrected values (transpose just gets the dims in the
    # right order)
    ds['S'] = S.transpose('time', 'depth')

    print("Mooring data corrected")
    return ds


def append_gsw_vars(ds, pref: float = 0):
    """Adds gsw variables like absolute salinity (SA), conservative
    temperature (CT), pressure calculated from nominal depth (P),
    and potential temperature (pt). Can specify the reference pressure,
    which is something to test w/r/t the initial conditions.

    Parameters
    --------
        pref : float, default 0
            reference pressure used in calculating potential density
    """
    print("Adding GSW variables to the mooring time series")
    lon, lat = -27.0048, -69.0005
    ds['p_from_z'] = gsw.p_from_z(ds['depth'], lat)
    ds['SA'] = gsw.SA_from_SP(ds['S'], ds['p_from_z'], lon, lat)
    ds['CT'] = gsw.CT_from_t(ds['SA'], ds['T'], ds['p_from_z'])
    ds['pt'] = gsw.pt_from_t(ds['SA'], ds['T'], ds['p_from_z'], pref)
    ds['pt'].attrs['reference pressure [dbar]'] = pref
    print("GSW variables added to the mooring time series")
    return ds


def plot_mooring_time_series(ds):

    # Init the plot
    cm = 1/2.54  # Inches to centimeters
    layout = [
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['ax1', 'ax1', 'ax1', 'ax2', 'ax2', 'ax2', 'ax3', 'ax3', 'ax3'],
        ['ax1', 'ax1', 'ax1', 'ax2', 'ax2', 'ax2', 'ax3', 'ax3', 'ax3'],
        ['ax1', 'ax1', 'ax1', 'ax2', 'ax2', 'ax2', 'ax3', 'ax3', 'ax3'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', '.'],
        ['ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', '.'],
        ['ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', '.'],
        ['ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', '.'],
        ['ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', '.']]
    fig, axd = plt.subplot_mosaic(layout)
    ax1, ax2, ax3 = axd['ax1'], axd['ax2'], axd['ax3']
    ax4, ax5 = axd['ax4'], axd['ax5']
    fig.set_figwidth(19*cm)
    fig.set_figheight(15*cm)

    # Add potential density
    ds['sigma0'] = gsw.sigma0(ds['SA'], ds['CT'])

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
    T_norm = plt.Normalize(T_min, T_max)  # TwoSlopeNorm(0, T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    sigma0_norm = plt.Normalize(sigma0_min, sigma0_max)
    dates = (datetime(2021, 9, 1, 0), datetime(2021, 10, 1, 0))
    T_ticks = [-1.8, -1.4, -1.0, -0.6, -0.2]
    T_labels = ['-1.8', '-1.4', '-1.0', '-0.6', '-0.2']
    S_ticks = [34.61, 34.67, 34.73, 34.79]
    S_labels = ['34.61', '34.67', '34.73', '34.79']
    sigma0_ticks = [27.735, 27.752, 27.769, 27.787]
    sigma0_labels = ['27.735', '27.752', '27.769', '27.787']

    def plotter(da, ax, norm, cmap, levels=150, add_colorbar=False):
        # For plotting the Hovmoellers
        # Not to self: Can't filter the turbulence because you loose
        # the visual of how deep the plume goes
        p1 = da.plot.contour(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar, cmap=cmap, zorder=1)
        p2 = da.plot.contourf(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar, cmap=cmap, zorder=1,
            rasterized=True)
        return p1, p2

    def spec_line(da, ax, value, minmax=False):
        # Add a specific line to show plume behavious
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
    p1a, p1b = plotter(da, ax1, T_norm, T_cmap)
    spec_line(da, ax1, T_min, minmax=True)
    spec_line(da, ax1, -1.3)
    spec_line(da, ax1, T_max, minmax=True)
    ax1.set_xlim(dates)

    # ax2: salinity
    da = ds['SA'].sel(depth=[-50, -125, -220])
    p2a, p2b = plotter(da, ax2, S_norm, S_cmap)
    spec_line(da, ax2, S_min, minmax=True)
    spec_line(da, ax2, 34.65)
    spec_line(da, ax2, S_max, minmax=True)
    ax2.set_xlim(dates)

    # ax3: potential density
    da = ds['sigma0'].sel(depth=[-50, -125, -220])
    p3a, p3b = plotter(da, ax3, sigma0_norm, sigma0_cmap)
    spec_line(da, ax3, sigma0_min, minmax=True)
    spec_line(da, ax3, 27.75)
    spec_line(da, ax3, sigma0_max, minmax=True)
    ax3.set_xlim(dates)

    # Filter wrapper
    def savgol_plotter(ds, var, d, ax, deriv=1, window=36, po=3, delta=2/24):
        # For plotting ROCs
        cmap_query = {-50: '#332288', -90: '#88CCEE', -125: "#328073",
                      -170: "#164626", -220: '#DDCC77', -250: '#AA4499'}
        da = ds[var].sel(depth=d)
        da_filtered = savgol_filter(
            da.values, window_length=window, polyorder=po, deriv=deriv,
            delta=delta)
        new_da = xr.DataArray(da_filtered, {'time': da['time']})
        new_da = new_da.sel(
            time=slice(datetime(2021, 9, 1), datetime(2021, 9, 21)))
        colour = cmap_query[d]
        p, = new_da.plot(ax=ax, c=colour, lw=1, label=d)
        ax.set_xlim(datetime(2021, 9, 1), datetime(2021, 9, 21))
        return p

    # ax4: temperature ROC
    pT5 = savgol_plotter(ds, 'T', -220, ax4)
    label5 = "220 m (T, S)"
    pT4 = savgol_plotter(ds, 'T', -170, ax4)
    label4 = "170 m (T)"
    pT3 = savgol_plotter(ds, 'T', -125, ax4)
    label3 = "125 m (T, S)"
    pT2 = savgol_plotter(ds, 'T', -90, ax4)
    label2 = "90 m (T)"
    pT1 = savgol_plotter(ds, 'T', -50, ax4)
    label1 = "50 m (T, S)"

    # ax5: salinity ROC
    pS3 = savgol_plotter(ds, 'SA', -220, ax5)
    pS2 = savgol_plotter(ds, 'SA', -125, ax5)
    pS1 = savgol_plotter(ds, 'SA', -50, ax5)

    # Hide unnecessary axis labels and add grid and ticks
    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(zorder=3)
        ax.set_ylim(-220, -50)
    ax1.set_yticks([-50, -90, -125, -170, -220])
    ax1.set_yticklabels(["50 m", "90 m", "125 m", "170 m", "220 m"])
    for ax in [ax2, ax3]:
        ax.set_yticks([-50, -125, -220])
        ax.set_yticklabels(["", "", ""])  # ["50 m", "125 m", "220 m"]
    for ax in [ax4, ax5]:
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
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([
            datetime(2021, 9, 1), datetime(2021, 9, 7), datetime(2021, 9, 14),
            datetime(2021, 9, 21), datetime(2021, 9, 28)])
        ax.set_xticklabels(['Sep', '7', '14', '21', '28'])

    # Add title annotations
    ax4.set_title(dTdt_title + ' (' + dTdt_units + ')', fontsize=12)
    ax5.set_title(dSdt_title + ' (' + dSdt_units + ')', fontsize=12)

    # Colourbars
    def mk_cbar(ax, p, units, xticks, xticklabels):
        cax = ax.inset_axes([0.05, 1.3, 0.9, 0.08], zorder=400)
        c = fig.colorbar(
            p, cax, orientation='horizontal', extend='both',
            format=ticker.FormatStrFormatter('%.2f'))
        c.ax.set_title(units, rotation=0, fontsize=12)
        c.ax.set_xticks(xticks)
        c.ax.set_xticklabels(xticklabels)
        c.ax.tick_params(labelsize=9, rotation=0)
    mk_cbar(ax1, p1a, T_title + ' (' + T_units + ')', T_ticks, T_labels)
    mk_cbar(ax2, p2a, S_title + ' (' + S_units + ')', S_ticks, S_labels)
    mk_cbar(ax3, p3a, sigma0_title + ' (' + sigma0_units + ')',
            sigma0_ticks, sigma0_labels)

    # Legend
    handles = [pT1, pT2, pT3, pT4, pT5]
    labels = [label1, label2, label3, label4, label5]
    ax4.legend(
        handles, labels, loc='center', bbox_to_anchor=(1.15, -0.3),
        bbox_transform=ax4.transAxes,
        title='Sensor level', frameon=False, fontsize=9)

    # Add important vertical lines
    for ax in [ax1, ax2, ax3]:
        ax.vlines(
            datetime(2021, 9, 5), ymin=-250, ymax=-50, colors='hotpink',
            lw=0.8)
        ax.vlines(
            datetime(2021, 9, 12), ymin=-250, ymax=-50, colors='hotpink',
            lw=0.8, ls='dashed')
    ax4.vlines(
        datetime(2021, 9, 5), ymin=-0.9, ymax=0.6, colors='hotpink', lw=0.8)
    ax4.vlines(
        datetime(2021, 9, 12), ymin=-0.9, ymax=0.6, colors='hotpink', lw=0.8,
        ls='dashed')
    ax5.vlines(
        datetime(2021, 9, 5), ymin=-0.08, ymax=0.05, colors='hotpink', lw=0.8)
    ax5.vlines(
        datetime(2021, 9, 12), ymin=-0.08, ymax=0.05, colors='hotpink',
        lw=0.8, ls='dashed')
    ax4.set_ylim(-0.8, 0.5)
    ax5.set_ylim(-0.08, 0.045)

    # Letter annotations
    bb = dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.1')
    labs = {ax1: 'a', ax2: 'b', ax3: 'c', ax4: 'd', ax5: 'e'}
    for ax in [ax1, ax2, ax3]:
        ax.text(
            0.08, 0.9, labs[ax], transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left', bbox=bb, zorder=120)
    for ax in [ax4, ax5]:
        ax.text(0.02, 0.9, labs[ax], transform=ax.transAxes, fontsize=12,
        fontweight='bold', va='top', ha='left', bbox=bb, zorder=120)

    # Denote the vertical line
    ax5.annotate(
        "Wind and air temp.\nanomaly (5 Sep)",
        xytext=(datetime(2021, 9, 5), -0.14),
        xy=(datetime(2021, 9, 5), -0.07), fontsize=9, c='hotpink',
        ha='left', arrowprops=dict(arrowstyle="->", color='hotpink'))
    ax5.annotate(
        "Air temp.\nanomaly (12 Sep)",
        xytext=(datetime(2021, 9, 10, 12), -0.14),
        xy=(datetime(2021, 9, 12), -0.07), fontsize=9, c='hotpink',
        ha='left', arrowprops=dict(arrowstyle="->", color='hotpink'))

    # Adjust spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.7, right=0.88)

    # Saving
    plt.savefig('figure_mooring.svg', transparent=False, dpi=600)


if __name__ == "__main__":
    ds = open_mooring_data()
    ds = correct_mooring_salinities(ds)
    ds = append_gsw_vars(ds)
    plot_mooring_time_series(ds)
