# For looking at atmospheric reanalyses near the mooring (ERA5) and sea ice
# maps from the AWI Sea Ice Portal

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
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
    for n,fp in enumerate(filepaths):
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


def select_nearest_coord(latitude, longitude):
    """Pass in the latitude (-40,-90) and longitude (-180,180) where you
    want the ice concentration. Returns the nearest x and y indices.
    Uses Haversine (assumes spherical Earth) so not accurage for large
    distances. There's probably a simpler way to do this but I already
    had this function."""

    with open('../filepaths/sea_ice_concentration') as f:
        dirpath = f.readlines()[0][:-1]  # Obscure the full filepath
    lonLat_Ant_fp = dirpath + '/LongitudeLatitudeGrid-s6250-Antarctic.hdf'
    lonLat_Arc_fp = dirpath + '/LongitudeLatitudeGrid-n6250-Arctic.hdf'
    lonLat_Ant_hdf = SD(lonLat_Ant_fp, SDC.READ)
    lonLat_Arc_hdf = SD(lonLat_Arc_fp, SDC.READ)
    lon_Ant_data = lonLat_Ant_hdf.select('Longitudes').get()
    lon_Arc_data = lonLat_Arc_hdf.select('Longitudes').get()
    lat_Ant_data = lonLat_Ant_hdf.select('Latitudes').get()
    lat_Arc_data = lonLat_Arc_hdf.select('Latitudes').get()
    lonLat_Ant_hdf.end()
    lonLat_Arc_hdf.end()

    # Because the grid from AWI is 0-360
    if longitude<0: longitude = longitude + 360

    # Credit: (https://stackoverflow.com/questions/69556412/
    #          with-a-dataframe-that-contains-coordinates-find-other-
    #          rows-with-coordinates-wit)
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1 = np.radians(lon1), np.radians(lat1)
        lon2, lat2 = np.radians(lon2), np.radians(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        haver_formula = (np.sin(dlat/2)**2 + np.cos(lat1) *
                         np.cos(lat2) * np.sin(dlon/2)**2)
        r = 3958.756  # 6371 for distance in KM for miles use 3958.756
        dist = 2 * r * np.arcsin(np.sqrt(haver_formula))
        return dist
    distances = haversine(lon_Ant_data, lat_Ant_data, longitude, latitude)
    id = np.where(distances == distances.min())

    return id


def plot_atm_and_ice():
    """Make figure for paper."""

    # Open the ERA5 data
    with open('../filepaths/ERA5') as f:  # Obscure the filepaths
        dir_fp = f.readlines()[0][:-1]
    file_temp = '/ERA5_mooring/ERA5_mooring_t2m_sp.nc'
    file_wind = '/ERA5_mooring/ERA5_mooring_u10_v10.nc'
    file_fp_temp = dir_fp + file_temp
    file_fp_wind = dir_fp + file_wind
    ds_temp = xr.open_dataset(file_fp_temp)
    ds_wind = xr.open_dataset(file_fp_wind)

    # Open the AWI ice portal sea ice data
    # Was previously pre-processed using Python
    with open('../filepaths/sea_ice_concentration') as f:
        dirpath = f.readlines()[0][:-1]
    filepath = dirpath + '/sea_ice_concentration.nc'

    # For the maps
    ds = xr.open_dataset(filepath)
    dateranges = [["20210401", "20210701"], ["20210701", "20211001"],
                  ["20211001", "20220101"], ["20220101", "20220401"],
                  ["20210905", "20210905"]]
    titles = [["Apr-Jun 2021"], ["Jul-Sep 2021"], ["Oct-Dec 2021"],
              ["Jan-Mar 2022"], ["5 Sep 2021"]]
    
    # For the time series
    id = select_nearest_coord(longitude=-27.0048333, latitude=-69.0005000)
    da_si = xr.open_dataset(filepath)['ice_conc']
    da_si = da_si.isel(x=id[0], y=id[1], drop=True)

    # Extract the ERA5 data that we want
    t2m = ds_temp['t2m'].interp(longitude=-27.0048, latitude=-69.0005)-273.15
    eastward_wind = ds_wind['u10'].interp(
        longitude=-27.0048, latitude=-69.0005)
    northward_wind = ds_wind['v10'].interp(
        longitude=-27.0048, latitude=-69.0005)
    wind = (eastward_wind**2 + northward_wind**2)**0.5

    # Plotting
    cm = 1/2.54  # Inches to centimeters
    layout = [['ax1', 'ax1', 'ax1', 'ax1', 'ax1', 'ax1', 'ax8', 'ax8'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax1', 'ax1', 'ax8', 'ax8'],
              ['ax2', 'ax2', 'ax2', 'ax2', 'ax2', 'ax2', 'ax8', 'ax8'],
              ['ax2', 'ax2', 'ax2', 'ax2', 'ax2', 'ax2', 'ax8', 'ax8'],
              ['ax7', 'ax7', 'ax7', 'ax7', 'ax7', 'ax7', 'ax8', 'ax8'],
              ['ax7', 'ax7', 'ax7', 'ax7', 'ax7', 'ax7', 'ax8', 'ax8'],
              ['.', '.', '.', '.', '.', '.', '.', '.'],
              ['ax3', 'ax3', 'ax4', 'ax4', 'ax5', 'ax5', 'ax6', 'ax6'],
              ['ax3', 'ax3', 'ax4', 'ax4', 'ax5', 'ax5', 'ax6', 'ax6'],
              ['ax3', 'ax3', 'ax4', 'ax4', 'ax5', 'ax5', 'ax6', 'ax6'],
              ['ax3', 'ax3', 'ax4', 'ax4', 'ax5', 'ax5', 'ax6', 'ax6'],
              ['.', '.', '.', '.', '.', '.', '.', '.']]
    proj = ccrs.Mercator(central_longitude=-69,
                            min_latitude=-80, max_latitude=-50, 
                            latitude_true_scale=-69)
    subplot_kw = dict(projection=proj)
    fig, axd = plt.subplot_mosaic(
        layout, per_subplot_kw={("ax3", "ax4", "ax5", "ax6", "ax8"):
                                subplot_kw})
    fig.set_figwidth(19*cm)
    fig.set_figheight(15*cm)
    ax1, ax2, ax7, ax8 = axd['ax1'], axd['ax2'], axd['ax7'], axd['ax8']
    ax3, ax4, ax5, ax6 = axd['ax3'], axd['ax4'], axd['ax5'], axd['ax6']

    # Plot wind
    t2m_plot = t2m.resample(valid_time="d").mean()
    wind_plot = wind.resample(valid_time="d").mean()
    t2m_plot.plot(ax=ax1, c='k', zorder=100, lw=1)
    wind_plot.plot(ax=ax2, c='k', zorder=100, lw=1)

    # Plot sea ice time series
    da_si.plot(ax=ax7, c='k', zorder=100, lw=1)

    # Denote forcing date and add max and min points
    forcing_date = dt.strptime('2021-09-05', '%Y-%m-%d')
    awi_c = (55/256, 167/256, 222/256)
    other_c = 'hotpink'
    ax1.vlines(x=forcing_date, ymin=-35, ymax=2, colors=other_c, lw=0.8)
    ax2.vlines(x=forcing_date, ymin=0, ymax=25, colors=other_c, lw=0.8)
    ax7.vlines(x=forcing_date, ymin=0, ymax=100, colors=other_c, lw=0.8)
    t2m_min = t2m_plot.sel(valid_time=forcing_date)
    ax1.scatter(forcing_date, t2m_min, s=100, c=awi_c, edgecolors='k',
                marker='.', lw=0.5, zorder=80)
    ax1.annotate(str(t2m_min.values)[:6]+" $℃$ (5 Sep)",
                 xytext=(forcing_date+td(days=25), -26), c=awi_c,
                 xy=(forcing_date, t2m_min), va="center", ha='left',
                 fontsize=9, arrowprops=dict(arrowstyle="->", color = awi_c))
    wind_max = wind_plot.sel(valid_time=forcing_date)
    ax2.scatter(forcing_date, wind_max, s=100, c=awi_c, edgecolors='k',
                marker='.', lw=0.5, zorder=80)
    ax2.annotate(str(wind_max.values)[:5]+" $m$ $s^{-1}$ (5 Sep)",
                 xytext=(forcing_date+td(days=25), 17), c=awi_c,
                 xy=(forcing_date, wind_max), va="center", ha='left',
                 fontsize=9, arrowprops=dict(arrowstyle="->", color = awi_c))

    # Denote the vertical line
    ax7.annotate("Atmos. forcing\nanomaly (5 Sep)",
                 xytext=(forcing_date-td(days=20), 15),
                 xy=(forcing_date, 10), fontsize=9, c=other_c, ha='right',
                 arrowprops=dict(arrowstyle="->", color = other_c))

    # For the sea ice concentration, it is actually min'ed 1 day later
    forcing_date = dt.strptime('2021-09-06', '%Y-%m-%d')
    min_si_conc = da_si.sel(date=forcing_date)
    ax7.scatter(forcing_date, min_si_conc, s=100, c=awi_c, edgecolors='k',
                marker='.', lw=0.5, zorder=80)
    ax7.annotate(str(min_si_conc.values[0][0])[:5]+"$\%$ (6 Sep)",
                 xytext=(forcing_date+td(days=15), 55),
                 xy=(forcing_date, min_si_conc), c=awi_c,
                 fontsize=9, arrowprops=dict(arrowstyle="->", color = awi_c))

    # Fix labels
    for ax in [ax1, ax2, ax7]:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax1.text(-0.16, 0.5, '$T_{2\ m}$\n($℃$)', ha='center', va='center',
             fontsize=12, transform=ax1.transAxes, zorder=110)
    ax2.text(-0.16, 0.5, '$U_{10\ m}$\n($m$ $s^{-1}$)', ha='center',
             va='center', fontsize=12, transform=ax2.transAxes, zorder=110)
    ax7.text(-0.16, 0.5, 'Sea\nice\nconc.\n($\%$)', ha='center', va='center',
             fontsize=12, transform=ax7.transAxes, zorder=110)

    # Fix ticks
    ax1.tick_params(axis='both', labelsize=9)
    ax2.tick_params(axis='both', labelsize=9)
    ax7.tick_params(axis='both', labelsize=9)
    ax1.tick_params(bottom=False, top=False, labelbottom=False)
    ax2.tick_params(bottom=False, top=False, labelbottom=False)

    # Control axis limits
    ax1.set_ylim(-35, 2)
    ax2.set_ylim(0, 20)
    ax7.set_ylim(0, 100)
    ax1.set_xlim(dt.strptime('2021-04-01', '%Y-%m-%d'),
                 dt.strptime('2022-04-01', '%Y-%m-%d'))
    ax2.set_xlim(dt.strptime('2021-04-01', '%Y-%m-%d'),
                 dt.strptime('2022-04-01', '%Y-%m-%d'))
    ax7.set_xlim(dt.strptime('2021-04-01', '%Y-%m-%d'),
                 dt.strptime('2022-04-01', '%Y-%m-%d'))

    # Grids (for the time series)
    ax1.grid()
    ax2.grid()
    ax7.grid()          

    # Plot ice
    land_50m = feature.NaturalEarthFeature(
        'physical', 'land', '50m', edgecolor='black', facecolor='white')
    for i, ax in enumerate([ax3, ax4, ax5, ax6, ax8]):
        ax.add_feature(land_50m, color='w')
        ax.coastlines(resolution='50m')
        ds_tmp = ds.sel(
            date=slice(dt.strptime(dateranges[i][0], '%Y%m%d'),
                       dt.strptime(dateranges[i][1], '%Y%m%d')))
        ds_tmp = ds_tmp.mean(dim='date')
        da = ds_tmp['ice_conc']
        c = ax.pcolormesh(da['lon'], da['lat'], da, cmap=cmocean.cm.ice,
            transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_title(titles[i][0], fontsize=12)#, pad=-0.1)
        ax.set_extent([-80, 0, -80, -50], crs=ccrs.PlateCarree())

        # Adding grid lines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                          y_inline=False, rotate_labels=False)
        gl.ylocator = mticker.FixedLocator([-80, -75, -70, -65, -60, -55, 50])
        gl.xlocator = mticker.FixedLocator([-100, -80, -60, -40, -20, 0])
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        gl.top_labels = False
        if ax==ax3 or ax==ax4 or ax==ax5:
            gl.right_labels=False
        if ax==ax4 or ax==ax5 or ax==ax6 or ax==ax8:
            gl.left_labels=False

        # Mark mooring location
        ax.scatter(-27.0048333, -69.0005000, s=100, c='w',
                   edgecolors='k', marker='*', lw=0.8,
                   transform=ccrs.PlateCarree(), zorder=130,
                   label='Mooring\n27.0° W\n69.0° S')

    # Change border colour
    ax8.tick_params(color=other_c, labelcolor=other_c)
    for spine in ax8.spines.values():
        spine.set_edgecolor(other_c)

    # Add sea ice concentration colourbar
    cbar_ax = fig.add_axes([0.23, 0.1, 0.45, 0.02])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_xlabel('Sea ice concentration ($\%$)',
                       fontdict={'fontsize': '9'})

    # Add legend for mooring symbol
    ax6.legend(edgecolor='white', prop={'size': '9'}, handletextpad=0.08,
              bbox_to_anchor=[0.3, -0.4], loc='center')

    # Annotate panel letters
    bb = dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.1')
    labs = {ax1: 'a', ax2: 'b', ax7: 'c', ax8: 'd', ax3: 'e', ax4: 'f',
            ax5: 'g', ax6: 'h'}
    for ax in [ax1, ax2, ax7]:
        ax.text(0.02, 0.9, labs[ax], transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left', bbox=bb, zorder=120)
    for ax in [ax3, ax4, ax5, ax6, ax8]:
        ax.text(0.08, 0.95, labs[ax], transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left', bbox=bb, zorder=120)

    # Final stuff and saving
    plt.subplots_adjust(top=0.92, right=0.92, left=0.15, bottom=0.1,
                        hspace=0.35, wspace=0.15)
    plt.savefig('figure_mooring_atm_and_ice.svg', transparent=False)


if __name__ == "__main__":
    # Example usage:
    # sea_ice_conc_nc('20210326', '20220501')
    plot_atm_and_ice()
