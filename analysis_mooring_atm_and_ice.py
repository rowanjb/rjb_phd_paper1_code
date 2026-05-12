# For looking at atmospheric reanalyses near the mooring (ERA5) and sea ice
# maps from the AWI Sea Ice Portal. The ERA5 data was downloaded using
# the CDS API Python package (for example usage, see: cds_request.py).
# The ice portal data is specifically from the University of Bremen’s
# Institute of Environmental Physics AMSR2 dataset (Spreen et al., 2008).
# I accessed it from the ice portal's https server on January 16, 2025.
# I /think/ I used wget for this, but in any case it can be done easily
# from a browser. The ERA5 data comes as netcdfs whereas the ice portal
# data comes as .hdf files, and these require processing using the
# sea_ice_conc_nc() function.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime as dt
from datetime import timedelta as td
import cartopy.crs as ccrs
import cartopy.feature as feature
from pyhdf.SD import SD, SDC
import cmocean


def list_of_date_strs(start_date_str, end_date_str):  # Anachronism
    """Creates a list of dates (strings) between two given dates."""
    start_date = dt.strptime(start_date_str, '%Y%m%d')
    end_date = dt.strptime(end_date_str, '%Y%m%d')
    all_dates = [start_date + td(days=x) for x in
                 range((end_date-start_date).days + 1)]
    all_dates_str = [date.strftime('%Y%m%d') for date in all_dates]
    return all_dates_str, all_dates


def sea_ice_conc_nc(start_date_str, end_date_str):
    """Creates .nc of daily sea ice concentration from AWI ice portal
    .hdf files."""

    # Run download_data(date_str) in a loop between two dates
    all_dates_str, all_dates = list_of_date_strs(start_date_str, end_date_str)

    # Create a list of all the .hdf files
    # "sea_ice_concentration" contains the fp of where I store the data
    with open('../filepaths/sea_ice_concentration') as f:
        dirpath = f.readlines()[0][:-1]
    filepaths = [dirpath + '/concentration_data/' + 'asi-AMSR2-s6250-' +
                 date_str + '-v5.4.hdf' for date_str in all_dates_str]

    # Open the grid and mask files 
    landmask_Ant_fp = dirpath + '/landmask_Ant_6.25km.hdf'
    landmask_Arc_fp = dirpath + '/landmask_Arc_6.25km.hdf'
    lonLat_Ant_fp = dirpath + '/LongitudeLatitudeGrid-s6250-Antarctic.hdf'
    lonLat_Arc_fp = dirpath + '/LongitudeLatitudeGrid-n6250-Arctic.hdf'
    landmask_Ant_hdf = SD(landmask_Ant_fp, SDC.READ)
    landmask_Arc_hdf = SD(landmask_Arc_fp, SDC.READ)
    lonLat_Ant_hdf = SD(lonLat_Ant_fp, SDC.READ)
    lonLat_Arc_hdf = SD(lonLat_Arc_fp, SDC.READ)
    landmask_Ant_data = landmask_Ant_hdf.select('landmask Ant 6.25 km').get()
    landmask_Arc_data = landmask_Arc_hdf.select('landmask Arc 6.25 km').get()
    lon_Ant_data = lonLat_Ant_hdf.select('Longitudes').get()
    lon_Arc_data = lonLat_Arc_hdf.select('Longitudes').get()
    lat_Ant_data = lonLat_Ant_hdf.select('Latitudes').get()
    lat_Arc_data = lonLat_Arc_hdf.select('Latitudes').get()
    lonLat_Ant_hdf.end()
    lonLat_Arc_hdf.end()
    landmask_Ant_hdf.end()
    landmask_Arc_hdf.end()

    # Init a dataset with the coordinates but no variables
    desc = 'Ice concentration in Antarctic from the AWI sea ice portal'
    hist = "Created by Rowan Brown, 21.01.2025"
    url = 'https://data.meereisportal.de/relaunch/concentration?lang=en'
    ds = xr.Dataset(
        data_vars=dict(),
        coords=dict(
            lon=(['x', 'y'], lon_Ant_data),
            lat=(['x', 'y'], lat_Ant_data),
            mask = (['x', 'y'], landmask_Ant_data)),
        attrs={'Description': desc,
               'History': hist,
               'URL:': url})

    # Loop through the .hdf files, create a dataset, and combine it with ds
    for n, fp in enumerate(filepaths):
        ice_hdf = SD(fp, SDC.READ)
        ice_data = ice_hdf.select('ASI Ice Concentration').get()
        ice_hdf.end()
        ice_ds = xr.Dataset(
            data_vars=dict(ice_conc = (['x', 'y'], ice_data)),
            coords=dict(
                lon=(['x', 'y'], lon_Ant_data),
                lat=(['x', 'y'], lat_Ant_data),
                date=('date', [all_dates[n]])))
        try:  # Concat won't work for the first .hdf
            ds = xr.concat([ds, ice_ds], dim='date')
        except:  # ...but merge will
            ds = xr.merge([ds, ice_ds])
        print(fp + ' added to .nc')

    ds.to_netcdf(dirpath + '/sea_ice_concentration.nc')
    print(ds)
    print('Ice concentration saved as .nc')


if __name__ == "__main__":
    # Example usage:
    sea_ice_conc_nc('20210326', '20220501')