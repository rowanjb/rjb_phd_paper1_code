# To make these plots you need to run analysis_mooring_time_series.py first

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
    if longitude < 0:
        longitude = longitude + 360

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


def plot_mooring_atm_and_ice():
    """Make figure for paper."""

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
    dateranges = [
        ["20210401", "20210701"], ["20210701", "20211001"],
        ["20211001", "20220101"], ["20220101", "20220401"],
        ["20210829", "20210829"], ["20210905", "20210905"],
        ["20210912", "20210912"]]
    titles = [
        ["(g) Apr-Jun 2021"], ["(h) Jul-Sep 2021"], ["(j) Oct-Dec 2021"],
        ["(j) Jan-Mar 2022"], ["(a) 29 Aug 2021"], ["(b) 5 Sep 2021"],
        ["(c) 12 Sep 2021"]]

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
    layout = [
        ['ax8','ax8','ax8', 'ax8','ax9', 'ax9', 'ax9','ax9', 'a10', 'a10', 'a10',  'a10',],
        ['ax8','ax8','ax8', 'ax8','ax9', 'ax9', 'ax9','ax9', 'a10', 'a10', 'a10',  'a10',],
        ['ax8','ax8','ax8', 'ax8','ax9', 'ax9', 'ax9','ax9', 'a10', 'a10', 'a10',  'a10',],
        ['ax8','ax8','ax8', 'ax8','ax9', 'ax9', 'ax9','ax9', 'a10', 'a10', 'a10',  'a10',],
        ['ax8','ax8','ax8', 'ax8','ax9', 'ax9', 'ax9','ax9', 'a10', 'a10', 'a10',  'a10',],
        [  '.',  '.',  '.',   '.',  '.',   '.',   '.',  '.',   '.',   '.',   '.',    '.',],
        [  '.',  '.',  '.',   '.',  '.',   '.',   '.',  '.',   '.',   '.',   '.',    '.',],
        [  '.','ax1','ax1', 'ax1','ax1', 'ax1', 'ax1','ax1', 'ax1', 'ax1', 'ax1',    '.',],
        [  '.','ax1','ax1', 'ax1','ax1', 'ax1', 'ax1','ax1', 'ax1', 'ax1', 'ax1',    '.',],
        [  '.','ax1','ax1', 'ax1','ax1', 'ax1', 'ax1','ax1', 'ax1', 'ax1', 'ax1',    '.',],
        [  '.','ax1','ax1', 'ax1','ax1', 'ax1', 'ax1','ax1', 'ax1', 'ax1', 'ax1',    '.',],
        [  '.','ax2','ax2', 'ax2','ax2', 'ax2', 'ax2','ax2', 'ax2', 'ax2', 'ax2',    '.',],
        [  '.','ax2','ax2', 'ax2','ax2', 'ax2', 'ax2','ax2', 'ax2', 'ax2', 'ax2',    '.',],
        [  '.','ax2','ax2', 'ax2','ax2', 'ax2', 'ax2','ax2', 'ax2', 'ax2', 'ax2',    '.',],
        [  '.','ax2','ax2', 'ax2','ax2', 'ax2', 'ax2','ax2', 'ax2', 'ax2', 'ax2',    '.',],
        [  '.','ax7','ax7', 'ax7','ax7', 'ax7', 'ax7','ax7', 'ax7', 'ax7', 'ax7',    '.',],
        [  '.','ax7','ax7', 'ax7','ax7', 'ax7', 'ax7','ax7', 'ax7', 'ax7', 'ax7',    '.',],
        [  '.','ax7','ax7', 'ax7','ax7', 'ax7', 'ax7','ax7', 'ax7', 'ax7', 'ax7',    '.',],
        [  '.','ax7','ax7', 'ax7','ax7', 'ax7', 'ax7','ax7', 'ax7', 'ax7', 'ax7',    '.',],
        [  '.',  '.',  '.',   '.',  '.',   '.',   '.',  '.',   '.',   '.',   '.',    '.',],
        [  '.',  '.',  '.',   '.',  '.',   '.',   '.',  '.',   '.',   '.',   '.',    '.',],
        ['ax3','ax3','ax3', 'ax4','ax4', 'ax4', 'ax5','ax5', 'ax5', 'ax6', 'ax6',  'ax6',],
        ['ax3','ax3','ax3', 'ax4','ax4', 'ax4', 'ax5','ax5', 'ax5', 'ax6', 'ax6',  'ax6',],
        ['ax3','ax3','ax3', 'ax4','ax4', 'ax4', 'ax5','ax5', 'ax5', 'ax6', 'ax6',  'ax6',],
        ['ax3','ax3','ax3', 'ax4','ax4', 'ax4', 'ax5','ax5', 'ax5', 'ax6', 'ax6',  'ax6',],
        ['ax3','ax3','ax3', 'ax4','ax4', 'ax4', 'ax5','ax5', 'ax5', 'ax6', 'ax6',  'ax6',],
        [  '.',  '.',  '.',   '.',  '.',   '.',   '.',  '.',   '.',   '.',   '.',    '.',],
        [  '.',  '.',  '.',   '.',  '.',   '.',   '.',  '.',   '.',   '.',   '.',    '.',]]
    proj = ccrs.Mercator(
        central_longitude=-69, min_latitude=-80, max_latitude=-50,
        latitude_true_scale=-69)
    subplot_kw = dict(projection=proj)
    fig, axd = plt.subplot_mosaic(
        layout, per_subplot_kw={
            ("ax3", "ax4", "ax5", "ax6", "ax8", "ax9", "a10"): subplot_kw})
    fig.set_figwidth(18*cm)
    fig.set_figheight(18*cm)
    ax1, ax2, ax7 = axd['ax1'], axd['ax2'], axd['ax7']
    ax3, ax4, ax5, ax6 = axd['ax3'], axd['ax4'], axd['ax5'], axd['ax6']
    ax8, ax9, ax10 = axd['ax8'], axd['ax9'], axd['a10']

    # Plot time series
    t2m_plot = t2m.resample(valid_time="d").mean()
    wind_plot = wind.resample(valid_time="d").mean()
    t2m_plot.plot(ax=ax1, c='k', zorder=100, lw=1)
    wind_plot.plot(ax=ax2, c='k', zorder=100, lw=1)
    da_si.plot(ax=ax7, c='k', zorder=100, lw=1)

    # Control spines
    for ax in [ax1, ax2, ax7]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Highlighting the sea ice map period
    for ax in [ax1, ax2, ax7]:
        rect = Rectangle(
            (dt(2021, 8, 29), -50), td(days=14),
            200, color="#e5e4e4")
        ax.add_patch(rect)

    # Fix labels
    for ax in [ax1, ax2, ax7]:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax1.text(
        -0.15, 0.6, '(d) 2 m air\ntemp. ($℃$)', fontsize=8,
        transform=ax1.transAxes, va='top', ha='center')
    ax2.text(
        -0.15, 0.6, '(e) 10 m air\nspeed (m s$^{-1}$)', fontsize=8,
        transform=ax2.transAxes, va='top', ha='center')
    ax7.text(
        -0.15, 0.6, '(f) Sea ice\nconc. (%)', fontsize=8,
        transform=ax7.transAxes, va='top', ha='center')

    # Fix ticks
    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax7.tick_params(axis='both', labelsize=8)
    ax1.tick_params(top=False, labelbottom=False)
    ax2.tick_params(top=False, labelbottom=False)

    # Control axis limits
    ax1.set_ylim(-35, 2)
    ax2.set_ylim(0, 20)
    ax7.set_ylim(-10, 120)
    ax7.set_yticks([0, 50, 100])
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

    # Control x ticks
    for ax in [ax1, ax2, ax7]:
        ax.set_xticks(
            [dt(2021, 4, 1), dt(2021, 5, 1), dt(2021, 6, 1), dt(2021, 7, 1),
             dt(2021, 8, 1), dt(2021, 9, 1), dt(2021, 10, 1), dt(2021, 11, 1),
             dt(2021, 12, 1), dt(2022, 1, 1), dt(2022, 2, 1), dt(2022, 3, 1),
             dt(2022, 4, 1)])

    # Plot ice
    land_50m = feature.NaturalEarthFeature(
        'physical', 'land', '50m', edgecolor='black', facecolor='white')
    for i, ax in enumerate([ax3, ax4, ax5, ax6, ax8, ax9, ax10]):
        ax.add_feature(land_50m, color='w')
        ax.coastlines(resolution='50m')
        ds_tmp = ds.sel(
            date=slice(dt.strptime(dateranges[i][0], '%Y%m%d'),
                       dt.strptime(dateranges[i][1], '%Y%m%d')))
        ds_tmp = ds_tmp.mean(dim='date')
        da = ds_tmp['ice_conc']
        c = ax.pcolormesh(da['lon'], da['lat'], da, cmap=cmocean.cm.ice,
            transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_title(titles[i][0], fontsize=8)
        #if ax==ax8 or ax==ax9 or ax==ax10:
        #    ax.set_extent([-100, 20, -95, -35], crs=ccrs.PlateCarree())
        #else:
        ax.set_extent([-80, 0, -75, -55], crs=ccrs.PlateCarree())

        # Adding grid lines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                          y_inline=False, rotate_labels=False)
        gl.ylocator = mticker.FixedLocator([-80, -75, -70, -65, -60, -55, 50])
        gl.xlocator = mticker.FixedLocator([-100, -80, -60, -40, -20, 0])
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.top_labels = False
        if ax==ax3 or ax==ax4 or ax==ax5 or ax==ax8 or ax==ax9:
            gl.right_labels=False
        if ax==ax4 or ax==ax5 or ax==ax6 or ax==ax9 or ax==ax10:
            gl.left_labels=False

        # Mark mooring location
        ax.scatter(
            -27.0048333, -69.0005000, s=100, c='w', edgecolors='k',
            marker='*', lw=0.8, transform=ccrs.PlateCarree(), zorder=130,
            label='Mooring\n27.0° W\n69.0° S')

    # Add sea ice concentration colourbar
    cbar_ax = fig.add_axes([0.23, 0.1, 0.45, 0.02])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('Sea ice concentration ($\%$)',
                       fontdict={'fontsize': '8'})

    # Add final lettering
    '''
    labs = {ax1: '(d)', ax2: '(e)', ax7: '(f)'}
    for ax in [ax1, ax2, ax7]:
        ax.text(
            -0.05, 1, labs[ax], transform=ax.transAxes, fontsize=8,
            va='top', ha='left', zorder=120)
    '''

    # Add legend for mooring symbol
    ax6.legend(edgecolor='white', prop={'size': '8'}, handletextpad=0.08,
              bbox_to_anchor=[0.3, -0.6], loc='center')

    # Final stuff and saving
    plt.subplots_adjust(top=0.95, right=0.94, left=0.11, bottom=0.1,
                        hspace=1, wspace=0.5)
    plt.savefig('figure_mooring_atm_and_ice.svg')
    plt.savefig('figure_mooring_atm_and_ice.pdf')
    plt.savefig('figure_mooring_atm_and_ice.png', dpi=300)


if __name__ == "__main__":
    plot_mooring_atm_and_ice()
