# Functions for opening, processing, and plotting data relating to the
# Weddell Sea mooring. Based on older functions and oop, but slimmed
# down here. Requires a few datasets:
#   (1) Sea ice concentration (see mooring_atm_and_ice.py for details)
#   (2) WOA data
#   (3) Mooring data
# Details to come!

import xarray as xr
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.io as spio
from scipy.interpolate import splrep, splev
import xarray as xr
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.patches as ptcs
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import cmocean


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
    """Opens the mooring .mat file(s)"""

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
    """ Unfortunately, the salinities from the lower two sensors seem,
    basically, wrong. I am waiting on word from the observationalists
    about how they calibrated/corrected the salinities originally, but
    until then I will use this function. I suspect they might not have
    had time to calibrate the sensors before launching them.
        Essentially, the lower two salinities from are ~2 PSU too
    fresh, which is unphysical. The corrected salinities from sal_BGC
    have no documentation, so in this function I'm basically overwriting
    the salinities by equating the sensors' means with those of a WOA
    climatology. This has the effect of slightly freshening the lower two
    salinities and salinifying the upper; in other words, destabilising
    the water column. I'm not happy about this but c'est la vie, because
    I have few alternatives.
        Importantly, this has no bearing on the proof that we have a
    plume, since this is predicated on the ROC of the sensors and not
    their actually values. This does however have a bearing on the model
    initial stratification, and is something that needs to be tested and
    handled carefully.
        Note that one potential source of error, particularly with the
    upper sensor, is that it's not 100% clear if it was at 135 m or
    125 m, so I correct this too. (I also assume it wasn't calibrated,
    even though it is seemingly more accurate than the other two.)
        For info on WOA data, see the WORLD OCEAN ATLAS 2023
    Product Documentation."""

    print("Beginning to correct the mooring data")

    # Calculate the mean salinites
    S = ds['S']  # Extract as a dataarray for easy handling
    S_srfce_mean_mooring = S.sel(depth=-50).mean(dim='time').values
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
    S_srfce_mean_woa = woa_weighted_mean.interp(depth=50).values
    S_upper_mean_woa = woa_weighted_mean.interp(depth=125).values
    S_lower_mean_woa = woa_weighted_mean.interp(depth=220).values

    # "Correct" the mooring salinities
    S_srfce_mean_anomaly = S_srfce_mean_mooring-S_srfce_mean_woa
    S_upper_mean_anomaly = S_upper_mean_mooring-S_upper_mean_woa
    S_lower_mean_anomaly = S_lower_mean_mooring-S_lower_mean_woa
    S = xr.where(S['depth'] == -50, S.sel(depth=-50)-S_srfce_mean_anomaly, S)
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
    fig.set_figheight(25*cm)

    # Cutting nans at start so the spline works
    ds = ds.sel(time=slice(datetime(2021, 4, 1), datetime(2022, 3, 31)))

    # Smooth the data using a spline
    # Helped with the syntax by chatgpt :(
    def smooth(da, s=100):
        """da should be a 1D array of T or S.
        s is the spline's smoothing factor."""
        t = da['time'].values.astype(float)
        y = da.values
        t_normed = (t - t.min()) / (t.max() - t.min())
        spline = splrep(t_normed, y, s=s)  # Create the B spline
        y_smooth = splev(t_normed, spline)  # Call the B spline
        da = xr.DataArray(y_smooth, {'time': da['time']})
        return da

    # == Plotting == #
    def plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2):
        ds[var].sel(depth=d).plot(
            ax=ax, label=var+" (raw)")
        smooth(ds[var].sel(depth=d), s=s).plot(
            ax=ax, label=var+" (s="+str(s)+")")
        ax.set_ylim(ymin1, ymax1)
        ax.set_ylabel(var+" @ "+str(d)+" m")
        axb = ax.twinx()
        smooth(ds[var].sel(depth=d), s=s).diff(dim='time').plot(
            ax=axb, label=var+" diff (s="+str(s)+")", ls=":", c='orange')
        axb.set_ylim(ymin2, ymax2)
        ax.set_xlim(datetime(2021, 8, 1), datetime(2021, 9, 30))
        axb.set_xlim(datetime(2021, 8, 1), datetime(2021, 9, 30))
        ax.legend(loc="lower center")
        axb.legend(loc="lower left")

    ax, var, d, s = ax1, "CT", -50, 0.5
    ymin1, ymax1 = -1.83, -1.78
    ymin2, ymax2 = -0.002, 0.002
    plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2)

    ax, var, d, s = ax2, "CT", -125, 150
    ymin1, ymax1 = -1.83, 0.5
    ymin2, ymax2 = -0.05, 0.05
    plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2)

    ax, var, d, s = ax3, "CT", -220, 5
    ymin1, ymax1 = 0.6, 0.9
    ymin2, ymax2 = -0.05, 0.05
    plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2)

    ax, var, d, s = ax4, "SA", -50, 0.005
    ymin1, ymax1 = 34.595, 34.64
    ymin2, ymax2 = -0.0015, 0.0015
    plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2)

    ax, var, d, s = ax5, "SA", -125, 1.2
    ymin1, ymax1 = 34.595, 34.8
    ymin2, ymax2 = -0.0015, 0.0015
    plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2)

    ax, var, d, s = ax6, "SA", -220, 0.15
    ymin1, ymax1 = 34.84, 34.88
    ymin2, ymax2 = -0.0015, 0.0015
    plotter(ds, ax, var, d, s, ymin1, ymax1, ymin2, ymax2)

    # Saving
    plt.savefig('figure_mooring_smoothing.png')


if __name__ == "__main__":
    # ds = open_mooring_data()
    # ds = correct_mooring_salinities(ds)
    # ds = append_gsw_vars(ds)
    ds = xr.open_dataset('tmp.nc')  # Delete this later
    plot_mooring_ROCs(ds)

    # DONT FORGET TO TEST THE DS/DT THING WITH THE UNCORRECTED SALINITIES
    # ds.fill_mooring()
    # ds.convert_to_daily()
    # ds.append_gsw_vars()
    # ds = open_mooring_profiles_data()
