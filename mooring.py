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


def plot_mooring_ROCs(ds):
    """Plots the rate-of-change of the variables at the mooring.
    Intended to support the analysis methods used later in the paper."""

    # Init the plot
    cm = 1/2.54  # Inches to centimeters
    fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=6)
    fig.set_figwidth(19*cm)
    fig.set_figheight(33*cm)

    # Cutting nans at start so the spline works
    ds = ds.sel(time=slice(datetime(2021, 8, 15), datetime(2021, 9, 30)))

    # Filter wrapper
    def savgol(da, deriv, window=24, po=3, delta=7200):
        da_filtered = savgol_filter(
            da.values, window_length=window, polyorder=po, deriv=deriv,
            delta=delta)
        return xr.DataArray(da_filtered, {'time': da['time']})

    # == Plotting == #
    def plotter(ds, ax, var, d):
        da = ds[var].sel(depth=d)
        da.plot(ax=ax, label=var+" (raw)")
        savgol(da, deriv=0).plot(ax=ax, label=var+" (smoothed)")
        ax.set_ylabel(var+" @ "+str(d)+" m")
        axb = ax.twinx()
        savgol(da, deriv=1).plot(ax=axb, label=var+" diff (smoothed)",
                                 ls="-", c='orange')
        ax.set_xlim(datetime(2021, 8, 15), datetime(2021, 9, 30))
        axb.set_xlim(datetime(2021, 8, 15), datetime(2021, 9, 30))
        ax.legend(loc="upper right")
        axb.legend(loc="lower left")

    plotter(ds, ax=ax1, var="CT", d=-50)
    plotter(ds, ax=ax2, var="CT", d=-125)
    plotter(ds, ax=ax3, var="CT", d=-220)
    plotter(ds, ax=ax4, var="SA", d=-50)
    plotter(ds, ax=ax5, var="SA", d=-125)
    plotter(ds, ax=ax6, var="SA", d=-220)

    # Saving
    plt.savefig('figure_mooring_smoothing.png')


def plot_hovmoellers(ds):

    # Init the plot
    cm = 1/2.54  # Inches to centimeters
    layout = [
        ['ax1a', 'ax1a', 'ax1a', 'ax1a', 'ax1a', 'ax1b', 'ax1b', 'ax1b',
         'ax1b', 'ax1b', 'ax1c', 'ax1c', 'ax1c', 'ax1c', 'ax1c', 'ax1c', '.'],
        ['ax1a', 'ax1a', 'ax1a', 'ax1a', 'ax1a', 'ax1b', 'ax1b', 'ax1b',
         'ax1b', 'ax1b', 'ax1c', 'ax1c', 'ax1c', 'ax1c', 'ax1c', 'ax1c', '.'],
        ['ax1a', 'ax1a', 'ax1a', 'ax1a', 'ax1a', 'ax1b', 'ax1b', 'ax1b',
         'ax1b', 'ax1b', 'ax1c', 'ax1c', 'ax1c', 'ax1c', 'ax1c', 'ax1c', '.'],
        ['ax2a', 'ax2a', 'ax2a', 'ax2a', 'ax2a', 'ax2b', 'ax2b', 'ax2b',
         'ax2b', 'ax2b', 'ax2c', 'ax2c', 'ax2c', 'ax2c', 'ax2c', 'ax2c', '.'],
        ['ax2a', 'ax2a', 'ax2a', 'ax2a', 'ax2a', 'ax2b', 'ax2b', 'ax2b',
         'ax2b', 'ax2b', 'ax2c', 'ax2c', 'ax2c', 'ax2c', 'ax2c', 'ax2c', '.'],
        ['ax2a', 'ax2a', 'ax2a', 'ax2a', 'ax2a', 'ax2b', 'ax2b', 'ax2b',
         'ax2b', 'ax2b', 'ax2c', 'ax2c', 'ax2c', 'ax2c', 'ax2c', 'ax2c', '.'],
        ['ax3a', 'ax3a', 'ax3a', 'ax3a', 'ax3a', 'ax3b', 'ax3b', 'ax3b',
         'ax3b', 'ax3b', 'ax3c', 'ax3c', 'ax3c', 'ax3c', 'ax3c', 'ax3c', '.'],
        ['ax3a', 'ax3a', 'ax3a', 'ax3a', 'ax3a', 'ax3b', 'ax3b', 'ax3b',
         'ax3b', 'ax3b', 'ax3c', 'ax3c', 'ax3c', 'ax3c', 'ax3c', 'ax3c', '.'],
        ['ax3a', 'ax3a', 'ax3a', 'ax3a', 'ax3a', 'ax3b', 'ax3b', 'ax3b',
         'ax3b', 'ax3b', 'ax3c', 'ax3c', 'ax3c', 'ax3c', 'ax3c', 'ax3c', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.',
         '.', '.', '.', '.'],
        ['ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4',
         'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', '.', '.'],
        ['ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4',
         'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', '.', '.'],
        ['ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4',
         'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', 'ax4', '.', '.'],
        ['ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5',
         'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', '.', '.'],
        ['ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5',
         'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', '.', '.'],
        ['ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5',
         'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', 'ax5', '.', '.']]
    fig, axd = plt.subplot_mosaic(layout)
    ax1a, ax2a, ax3a = axd['ax1a'], axd['ax2a'], axd['ax3a']
    ax1b, ax2b, ax3b = axd['ax1b'], axd['ax2b'], axd['ax3b']
    ax1c, ax2c, ax3c = axd['ax1c'], axd['ax2c'], axd['ax3c']
    ax4, ax5 = axd['ax4'], axd['ax5']
    fig.set_figwidth(19*cm)
    fig.set_figheight(19*cm)

    # Add potential density
    ds['sigma0'] = gsw.sigma0(ds['SA'], ds['CT'])

    # Colours and other misc
    T_min, T_max = -1.90, 1
    S_min, S_max = 34.4, 34.87
    sigma0_min, sigma0_max = 27.60, 27.82
    T_cmap = cmocean.cm.thermal
    S_cmap = cmocean.cm.haline
    sigma0_cmap = cmocean.cm.dense
    T_title = 'In-situ temp.'
    dTdt_title = "$dT/dt$ (in-situ temp.)"
    S_title = 'Absolute salinity'
    dSdt_title = "$dS/dt$ (absolute salinity)"
    sigma0_title = 'Pot. density ($\sigma_0$)'
    T_units = '$\degree C$'
    dTdt_units = '$℃$ $d^{-1}$'
    S_units = '$g$ $kg^{-1}$'
    dSdt_units = '$g$ $kg^{-1}$ $d^{-1}$'
    sigma0_units = '$kg$ $m^{-3}$'
    T_norm = TwoSlopeNorm(0, T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    sigma0_norm = plt.Normalize(sigma0_min, sigma0_max)
    dates_a = slice(datetime(2021, 4, 1, 0), datetime(2021, 9, 1, 0))
    dates_b = slice(datetime(2021, 9, 1, 0), datetime(2021, 10, 1, 0))
    dates_c = slice(datetime(2021, 10, 1, 0), datetime(2022, 4, 1, 0))

    def plotter(da, ax, norm, cmap, levels=150, add_colorbar=False):
        # Not to self: Can't filter the turbulence because you loose
        # the visual of how deep the plume goes
        p1 = da.plot.contour(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar,cmap=cmap, zorder=1)
        p2 = da.plot.contourf(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar, cmap=cmap, zorder=1)
        return p1, p2

    # ax1: temperature
    da = ds['T']
    p1a1, p1a2 = plotter(da.sel(time=dates_a), ax1a, T_norm, T_cmap)
    p1b1, p2b2 = plotter(da.sel(time=dates_b), ax1b, T_norm, T_cmap)
    p1c1, p1c2 = plotter(da.sel(time=dates_c), ax1c, T_norm, T_cmap)

    # ax2: salinity
    da = ds['SA'].sel(depth=[-50, -125, -220])
    p2a1, p2a2 = plotter(da.sel(time=dates_a), ax2a, S_norm, S_cmap)
    p2b2, p2b2 = plotter(da.sel(time=dates_b), ax2b, S_norm, S_cmap)
    p2c1, p2c2 = plotter(da.sel(time=dates_c), ax2c, S_norm, S_cmap)

    # ax3: potential density
    da = ds['sigma0'].sel(depth=[-50, -125, -220])
    p3a1, p3a2 = plotter(da.sel(time=dates_a), ax3a, sigma0_norm, sigma0_cmap)
    p3b1, p3b2 = plotter(da.sel(time=dates_b), ax3b, sigma0_norm, sigma0_cmap)
    p3c1, p3c2 = plotter(da.sel(time=dates_c), ax3c, sigma0_norm, sigma0_cmap)

    # Filter wrapper
    def savgol_plotter(ds, var, d, ax, deriv=1, window=36, po=3, delta=2/24):
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
    pT6 = savgol_plotter(ds, 'T', -250, ax4)
    label6 = "250 m (T)"
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

    # Match the ax2 and ax3 max depths with that of ax1
    for ax in [ax2a, ax2b, ax2c, ax3a, ax3b, ax3c]:
        ax.set_ylim(-250, -50)

    # Hide unnecessary axis labels and add grid
    for ax in [ax2a, ax2b, ax2c, ax3a, ax3b, ax3c]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(zorder=3)
        ax.set_yticks([-50, -125, -220])
        ax.set_yticklabels(["50 m", "125 m", "220 m"])
    for ax in [ax1a, ax1b, ax1c]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(zorder=3)
        ax.set_yticks([-50, -90, -125, -170, -220, -250])
        ax.set_yticklabels(["50 m", "90 m", "125 m", "170 m", "220 m",
                            "250 m"])
    for ax in [ax4, ax5]:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(zorder=3)
    ax5.set_xticks([
        datetime(2021, 9, 1), datetime(2021, 9, 3), datetime(2021, 9, 5),
        datetime(2021, 9, 7), datetime(2021, 9, 9), datetime(2021, 9, 11),
        datetime(2021, 9, 13), datetime(2021, 9, 15), datetime(2021, 9, 17),
        datetime(2021, 9, 19), datetime(2021, 9, 21)])
    ax5.set_xticklabels(
        ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21'])

    # Hide the spines and ticks
    for ax in [ax1b, ax1c, ax2b, ax2c]:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.tick_params(left=False, labelleft=False)
    for ax in [ax3b, ax3c]:
        ax.tick_params(left=False, labelleft=False)
        ax.tick_params(axis='x', which='both', labelsize=9)
    for ax in [ax1a, ax2a, ax4]:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', labelsize=9)
    for ax in [ax3a, ax5]:
        ax.tick_params(axis='y', which='both', labelsize=9)
        ax.tick_params(axis='x', which='both', labelsize=9)
    for ax in [ax1a, ax2a, ax3a]:
        ax.spines[['right']].set_visible(False)
    for ax in [ax1b, ax2b, ax3b]:
        ax.spines[['right', 'left']].set_visible(False)
    for ax in [ax1c, ax2c, ax3c]:
        ax.spines[['left']].set_visible(False)

    # Define x-ticks
    a_ticks = [datetime(2021, 4, 1), datetime(2021, 5, 1),
               datetime(2021, 6, 1), datetime(2021, 7, 1),
               datetime(2021, 8, 1)]
    a_labels = ["Apr", "May", "Jun", "Jul", "Aug"]
    b_ticks = [datetime(2021, 9, 1), datetime(2021, 9, 7),
               datetime(2021, 9, 14), datetime(2021, 9, 21),
               datetime(2021, 9, 28)]
    b_labels = ['01', '07', '14', '21', '28']
    c_ticks = [datetime(2021, 10, 1), datetime(2021, 11, 1),
               datetime(2021, 12, 1), datetime(2022, 1, 1),
               datetime(2022, 2, 1), datetime(2022, 3, 1)]
    c_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

    # Set x-ticks
    for ax in [ax1a, ax2a, ax3a]:
        ax.set_xticks(a_ticks)
        ax.set_xticklabels(a_labels)
    for ax in [ax1b, ax2b, ax3b]:
        ax.set_xticks(b_ticks)
        ax.set_xticklabels(b_labels)
    for ax in [ax1c, ax2c, ax3c]:
        ax.set_xticks(c_ticks)
        ax.set_xticklabels(c_labels)

    # New axis labels
    ax3a.set_xlabel("2021", fontsize=9)
    ax3b.set_xlabel("September 2021", fontsize=9)
    ax5.set_xlabel("September 2021", fontsize=9)
    ax3c.set_xlabel("2021 / 2022", fontsize=9)

    # Add title annotations
    bb = dict(facecolor='white', edgecolor='white',
              boxstyle='square,pad=0.2', alpha=0.75)
    ax1c.text(0.03, 0.05, T_title, va='bottom', ha='left',
              transform=ax1a.transAxes, fontsize=12, zorder=200, bbox=bb)
    ax2c.text(0.03, 0.05, S_title, va='bottom', ha='left',
              transform=ax2a.transAxes, fontsize=12, zorder=200, bbox=bb)
    ax3c.text(0.03, 0.05, sigma0_title, va='bottom', ha='left',
              transform=ax3a.transAxes, fontsize=12, zorder=200, bbox=bb)
    ax4.text(0.01, 0.05, dTdt_title, va='bottom', ha='left',
              transform=ax4.transAxes, fontsize=12, zorder=200, bbox=bb)
    ax5.text(0.01, 0.05, dSdt_title, va='bottom', ha='left',
              transform=ax5.transAxes, fontsize=12, zorder=200, bbox=bb)

    # Colourbars
    def mk_cbar(ax, p2, p1, units):
        cax = ax.inset_axes([1.05, 0.05, 0.08, 0.9], zorder=400)
        c = fig.colorbar(p2, cax, orientation='vertical',
                         format=ticker.FormatStrFormatter('%.2f'),
                         extend='both')
        c.set_label(units, rotation=90, fontsize=9)
        c.ax.tick_params(labelsize=9)
        c.locator = ticker.MaxNLocator(nbins=7)
        c.update_ticks()
    mk_cbar(ax1c, p1c2, p1c1, T_units)
    mk_cbar(ax2c, p2c2, p2c1, S_units)
    mk_cbar(ax3c, p3c2, p3c1, sigma0_units)

    # Axis labels
    ax4.text(-0.13, 0.5, dTdt_units, ha='center', va='center',
             transform=ax4.transAxes, fontsize=9, rotation=90)
    ax5.text(-0.13, 0.5, dSdt_units, ha='center', va='center',
             transform=ax5.transAxes, fontsize=9, rotation=90)

    # Legend
    handles = [pT1, pT2, pT3, pT4, pT5, pT6]
    labels = [label1, label2, label3, label4, label5, label6]
    ax4.legend(
        handles, labels,
        loc='center', bbox_to_anchor=(1.15, 0), bbox_transform=ax4.transAxes,
        title='Sensor level', frameon=False, fontsize=9)

    # Add important vertical lines
    for ax in [ax1b, ax2b, ax3b]:
        ax.vlines(datetime(2021, 9, 5), ymin=-250, ymax=-50, colors='hotpink',
                  lw=0.8)
        ax.vlines(datetime(2021, 9, 12), ymin=-250, ymax=-50, colors='hotpink',
                  lw=0.8, ls='dashed')
    ax4.vlines(datetime(2021, 9, 5), ymin=-0.9, ymax=0.6, colors='hotpink',
               lw=0.8)
    ax4.vlines(datetime(2021, 9, 12), ymin=-0.9, ymax=0.6, colors='hotpink',
               lw=0.8, ls='dashed')
    ax5.vlines(datetime(2021, 9, 5), ymin=-0.08, ymax=0.05, colors='hotpink',
               lw=0.8)
    ax5.vlines(datetime(2021, 9, 12), ymin=-0.08, ymax=0.05, colors='hotpink',
               lw=0.8, ls='dashed')
    ax4.set_ylim(-0.8, 0.5)
    ax5.set_ylim(-0.08, 0.045)

    # Letter annotations
    bb = dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.1')
    labs = {ax1a: 'a', ax2a: 'b', ax3a: 'c', ax4: 'd', ax5: 'e'}
    for ax in [ax1a, ax2a, ax3a]:
        ax.text(0.08, 0.9, labs[ax], transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left', bbox=bb, zorder=120)
    for ax in [ax4, ax5]:
        ax.text(0.02, 0.9, labs[ax], transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left', bbox=bb, zorder=120)

    # Adjust spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.7, right=0.88)

    # Saving
    plt.savefig('figure_hovmoellers.svg', transparent=False)


def plot_hovmoellers2(ds):

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
    T_min, T_max = -1.82, 0
    S_min, S_max = 34.6, 34.8
    sigma0_min, sigma0_max = 27.73, 27.80
    T_cmap = mpl.colormaps['Blues_r']
    S_cmap = mpl.colormaps['Oranges']
    sigma0_cmap = mpl.colormaps['Purples']
    T_title = 'In-situ temp.'
    dTdt_title = "$dT/dt$ (in-situ temp.)"
    S_title = 'Abs. salinity'
    dSdt_title = "$dS/dt$ (absolute salinity)"
    sigma0_title = 'Pot. dens., $\sigma_0$'
    T_units = '$\degree C$'
    dTdt_units = '$℃$ $d^{-1}$'
    S_units = '$g$ $kg^{-1}$'
    dSdt_units = '$g$ $kg^{-1}$ $d^{-1}$'
    sigma0_units = '$kg$ $m^{-3}$'
    T_norm = plt.Normalize(T_min, T_max)  # TwoSlopeNorm(0, T_min, T_max)
    S_norm = plt.Normalize(S_min, S_max)
    sigma0_norm = plt.Normalize(sigma0_min, sigma0_max)
    dates = slice(datetime(2021, 9, 1, 0), datetime(2021, 10, 1, 0))

    def plotter(da, ax, norm, cmap, levels=150, add_colorbar=False):
        # For plotting the Hovmoellers
        # Not to self: Can't filter the turbulence because you loose
        # the visual of how deep the plume goes
        p1 = da.plot.contour(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar, cmap=cmap, zorder=1)
        p2 = da.plot.contourf(
            'time', 'depth', ax=ax, levels=levels, norm=norm,
            add_colorbar=add_colorbar, cmap=cmap, zorder=1)
        return p1, p2

    def spec_line(da, ax, value):
        # Add a specific line to show plume behavious
        da = da.rolling(time=6).mean()
        da.plot.contour(
            'time', 'depth', ax=ax, levels=1, vmin=value, vmax=value,
            add_colorbar=False, cmap=mpl.colormaps['Greys'], zorder=2,
            linestyles='solid', linewidths=0.5)

    # ax1: temperature
    da = ds['T'].sel(depth=[-50, -125, -220], time=dates)
    p1a, p1b = plotter(da, ax1, T_norm, T_cmap)
    spec_line(da, ax1, -1.3)

    # ax2: salinity
    da = ds['SA'].sel(depth=[-50, -125, -220], time=dates)
    p2a, p2b = plotter(da, ax2, S_norm, S_cmap)
    spec_line(da, ax2, 34.65)

    # ax3: potential density
    da = ds['sigma0'].sel(depth=[-50, -125, -220], time=dates)
    p3a, p3b = plotter(da, ax3, sigma0_norm, sigma0_cmap)
    spec_line(da, ax3, 27.75)

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
    ax5.set_xticks([
        datetime(2021, 9, 1), datetime(2021, 9, 3), datetime(2021, 9, 5),
        datetime(2021, 9, 7), datetime(2021, 9, 9), datetime(2021, 9, 11),
        datetime(2021, 9, 13), datetime(2021, 9, 15), datetime(2021, 9, 17),
        datetime(2021, 9, 19), datetime(2021, 9, 21)])
    ax5.set_xticklabels(
        ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21'])
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([
            datetime(2021, 9, 1), datetime(2021, 9, 7), datetime(2021, 9, 14),
            datetime(2021, 9, 21), datetime(2021, 9, 28)])
        ax.set_xticklabels(['01', '07', '14', '21', '28'])

    # Add title annotations
    ax4.set_title(dTdt_title, fontsize=12)
    ax5.set_title(dSdt_title, fontsize=12)

    # Colourbars
    def mk_cbar(ax, p, units):
        cax = ax.inset_axes([0.05, 1.32, 0.9, 0.08], zorder=400)
        c = fig.colorbar(
            p, cax, orientation='horizontal', extend='both',
            format=ticker.FormatStrFormatter('%.2f'))
        c.ax.set_title(units, rotation=0, fontsize=12)
        c.ax.tick_params(labelsize=9, rotation=30)
        c.locator = ticker.MaxNLocator(nbins=7)
        c.update_ticks()
    mk_cbar(ax1, p1a, T_title + ' (' + T_units + ')')
    mk_cbar(ax2, p2a, S_title + ' (' + S_units + ')')
    mk_cbar(ax3, p3a, sigma0_title + ' (' + sigma0_units + ')')

    # Axis labels
    ax4.text(-0.13, 0.5, dTdt_units, ha='center', va='center',
             transform=ax4.transAxes, fontsize=9, rotation=90)
    ax5.text(-0.13, 0.5, dSdt_units, ha='center', va='center',
             transform=ax5.transAxes, fontsize=9, rotation=90)

    # Legend
    handles = [pT1, pT2, pT3, pT4, pT5]
    labels = [label1, label2, label3, label4, label5]
    ax4.legend(
        handles, labels,
        loc='center', bbox_to_anchor=(1.15, 0), bbox_transform=ax4.transAxes,
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

    # Adjust spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.7, right=0.88)

    # Saving
    plt.savefig('figure_hovmoellers.svg', transparent=False)


if __name__ == "__main__":
    #ds = open_mooring_data()
    #ds = correct_mooring_salinities(ds)
    #ds = append_gsw_vars(ds)
    #ds.to_netcdf("tmp_partial_corred.nc")
    ds = xr.open_dataset('tmp_partial_corred.nc')  # Delete this later
    plot_hovmoellers2(ds)

    # DONT FORGET TO TEST THE DS/DT THING WITH THE UNCORRECTED SALINITIES
    # ds.fill_mooring()
    # ds.convert_to_daily()
    # ds.append_gsw_vars()
    # ds = open_mooring_profiles_data()
