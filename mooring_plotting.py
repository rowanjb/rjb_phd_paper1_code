# Rowan Brown, 08.2025
# Munich, Germany
# These were previously part of mooring_analyses.py but I'm separating
# analysis from plotting

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
import matplotlib.gridspec as gridspec
from sea_ice_concentration import select_nearest_coord
import mooring as ma
import cmocean


def convective_resistance(ds,type='heat'):
    """Calculates convective resistance, i.e., how much heat or mass needs to be removed to cause homogenization of the water column. 
    Reference depth is taken as 220 m, since this is the bottom working salinity sensor. 
    Note: Convective resistence /assumes/ that none of the heat loss goes into creating sea ice; maybe see Wilson and M. for more on this. 
        --->How to deal with pack ice? Some HF will make or melt ice, other will go into the water...
    """
    pot_rho, ref_depth = ds['pot_rho'], 220 # Variables needed for calculating the convective resistance 
    pot_rho = pot_rho.where(pot_rho>0,drop=True) # Dropping whereever we had a temp but no salinity and tf no rho
    pot_rho = pot_rho.assign_coords(dz=('depth',[(-1)*(pot_rho['depth'][n].values-d) for n,d in enumerate(np.append([0],pot_rho['depth'].values)[:-1])])) # Adding the 0 to get 50 at the start of the list  (ds['depth'][n]-ds['depth'][n-1])
    term2 = pot_rho * pot_rho['dz']
    if type=='heat':
        convr = 9.81*(ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth')) # Unit ends up being J/m3... I think
    elif type=='mass':
        convr = (ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth')) # Unit ends up being kg/m2
    else:
        print('Need to choose type="heat" or "mass"')
    return convr


def density_flux(ds):
    """/Assumes through the surface/. Also neglects absolute surface (where dS is because) because lack of data.
    Desnity flux represents the difference in mass in the water column between the stratified case and homogenized case."""

    # We'll need dz later
    # Adding the 0 to get 50 at the start of the list  (ds['depth'][n]-ds['depth'][n-1])
    dz = lambda ds : [(-1)*(ds['depth'][n].values-d) for n,d in enumerate(np.append([0],ds['depth'].values)[:-1])] 
    
    # Density/mass anomaly
    pot_rho, ref_depth = ds['pot_rho'], 220 # Variables needed for calculating the convective resistance 
    pot_rho = pot_rho.where(pot_rho>0,drop=True) # Dropping whereever we had a temp but no salinity and tf no rho
    pot_rho = pot_rho.assign_coords(dz=('depth', dz(pot_rho)))
    term2 = pot_rho * pot_rho['dz']
    dens_flux = ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth') 

    # Heat content (anomaly?)
    refT, rho_0, C_p = -1.8, 1026, 3992 # Alternative: ds['Cp'] = gsw.cp_t_exact(ds['S'],ds['T'],ds['p_from_z'])  
    T = ds['T']
    T = T.assign_coords(dz=('depth', dz(T)))
    HC = rho_0 * C_p * 10**(-9) * ((T.sel(depth=slice(0,(-1)*ref_depth))-refT)*T['dz']).sum(dim='depth') # the 10^-9 makes the result GJ

    # Salt content
    S = ds['S'].where(ds['S']>0,drop=True)
    S = S.assign_coords(dz=('depth',dz(S)))
    SC = (S*S['dz']).sum(dim='depth')
    
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(7, 4))
    color = 'tab:blue'
    dens_flux.plot(ax=ax,color=color)
    ax.set_ylabel('Mass anomaly ($kg$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.tick_params(size=9)
    ax.set_ylim(0,20)
    ax.set_title('Density changes at the Weddell Sea mooring',fontsize=12)
    ax.tick_params(size=9,color=color)
    ax.yaxis.label.set_color(color=color)
    ax.tick_params(axis='y', colors=color)

    ax2 = ax.twinx()
    color = 'tab:red'
    HC.plot(ax=ax2,color=color)
    ax2.set_ylabel('Heat content ($GJ$)',fontsize=11,color=color)
    ax2.set_xlabel('',fontsize=11)
    ax2.tick_params(size=9,color=color)
    ax2.yaxis.label.set_color(color=color)
    ax2.tick_params(axis='y', colors=color)

    ax3 = ax.twinx()
    color = 'tab:green'
    SC.plot(ax=ax3,color=color)
    ax3.set_ylabel('Salt content ($g$ $kg^{-1}$)',fontsize=11,color=color)
    ax3.set_xlabel('',fontsize=11)
    ax3.tick_params(size=9,color=color)
    ax3.yaxis.label.set_color(color=color)
    ax3.tick_params(axis='y', colors=color)
    ax3.spines.right.set_position(("axes", 1.2))

    # Adding the sea ice data to the plot
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
    ds_si = xr.open_dataset(filepath).sel(date=slice("2021-03-26", "2022-04-06")).isel(x=id[0],y=id[1])
    ax4 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:grey'
    ax4.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax4.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.spines.right.set_position(("axes", 1.4))

    plt.savefig('Figures/Density_flux.png',bbox_inches='tight',dpi=250)

    return dens_flux


# plotting temperature
def plt_hovm_CELLO(ds):
    """*Plots used in my CELLO poster, created specifically for posterity.
    Creates Hovmöller plots."""

    start_date = datetime(2021, 4, 1, 0, 0, 0)
    end_date = datetime(2022, 4, 1, 0, 0, 0)

    ds['pot_rho'] = gsw.sigma0(ds['SA'], ds['CT'])

    # Some var-specific definitions
    depths = {'T': [-50, -90, -125, -170, -220],
              'SA': [-50, -125, -220],
              'pot_rho': [-50, -125, -220]}
    titles = {'T': 'In-situ\ntemperature',
              'SA': 'Absolute\nsalinity',
              'pot_rho': 'Potential\ndensity'}
    units = {'T': '$\degree C$',
             'SA': '$g$ $kg^{-1}$',
             'pot_rho': '$kg$ $m^{-3}$'}
    lims = {'T': (-2, 1), 'SA': (34.07, 34.91), 'pot_rho': (27.5, 27.87)}
    cm = {'T': cmocean.cm.thermal,
          'SA': cmocean.cm.haline,
          'pot_rho': cmocean.cm.dense}

    # == Plotting == #

    plt.rcParams["font.family"] = "serif"
    f, axs = plt.subplots(nrows=3, ncols=1, figsize=(11, 7), sharex=True)

    vars = ['T', 'SA', 'pot_rho']
    for n, var in enumerate(vars):  # We want this order for reasons
        var = vars[n]
        lower_lim, upper_lim = lims[var]
        if vars[n] == 'T':
            norm = TwoSlopeNorm(0, lower_lim, upper_lim)
        else:  # Mapping to the colourbar internal [0, 1]
            norm = plt.Normalize(lower_lim, upper_lim)

        gs = axs[n].get_gridspec()
        f.delaxes(axs[n])
        gs_new = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[n], width_ratios=[5, 6, 6])

        axa = f.add_subplot(gs_new[0, 0])
        axb = f.add_subplot(gs_new[0, 1])
        axc = f.add_subplot(gs_new[0, 2])

        date_ranges = [
            slice(start_date, datetime(2021, 9, 1, 0)),
            slice(datetime(2021, 9, 1, 0), datetime(2021, 9, 28, 0)),
            slice(datetime(2021, 9, 28, 0), end_date)
        ]

        p1 = ds[var].sel(depth=depths[var]).sel(
            time=date_ranges[0]).plot.contourf(
            'time', 'depth', ax=axa, levels=24, norm=norm,
            add_colorbar=False, cmap=cm[var],
            rasterized=True, zorder=1)
        p2 = ds[var].sel(depth=depths[var]).sel(
            time=date_ranges[1]).plot.contourf(
            'time', 'depth', ax=axb, levels=24, norm=norm,
            add_colorbar=False, cmap=cm[var],
            rasterized=True, zorder=1)
        p3 = ds[var].sel(depth=depths[var]).sel(
            time=date_ranges[2]).plot.contourf(
            'time', 'depth', ax=axc, levels=24, norm=norm,
            add_colorbar=False, cmap=cm[var],
            rasterized=True, zorder=1)

        axa.spines[['right']].set_visible(False)
        axb.spines[['left', 'right']].set_visible(False)
        axc.spines[['left']].set_visible(False)

        axa.set_ylabel('')
        axb.set_ylabel('')
        axc.set_ylabel('')

        axa.set_xlabel('')
        axb.set_xlabel('')
        axc.set_xlabel('')

        if n != 2:
            axa.tick_params(axis='x', which='both', labelbottom=False)
            axb.tick_params(axis='x', which='both', labelbottom=False)
            axc.tick_params(axis='x', which='both', labelbottom=False)
        else:
            axa.tick_params(axis='x', which='both', labelsize=16, rotation=30)
            axb.tick_params(axis='x', which='both', labelsize=16, rotation=30)
            axc.tick_params(axis='x', which='both', labelsize=16, rotation=30)

        # Adding the sea ice data to the plot
        with open('../filepaths/sea_ice_concentration') as file:
            dirpath = file.readlines()[0][:-1]
        filepath = dirpath + '/sea_ice_concentration.nc'
        id = select_nearest_coord(longitude=-27.0048333, latitude=-69.0005000)
        da_si = xr.open_dataset(filepath)['ice_conc']
        da_si = da_si.isel(x=id[0], y=id[1], drop=True)
        da_si = da_si.sel(date=slice(np.datetime64(start_date),
                                     np.datetime64(end_date)))
        c = 'brown'
        for nsi, a in enumerate([axa, axb, axc]):
            axsi = a.twinx()
            da = da_si.sel(date=date_ranges[nsi])
            da.plot(ax=axsi, color=c, linewidth=1)
            axsi.set_ylim(0, 100)
            axsi.spines[['left', 'right']].set_visible(False)
            axsi.set_ylabel('')
            axsi.set_xlabel('')
            axsi.set_title('')
            axsi.tick_params(axis='y', which='both',
                             left=False, labelleft=False,
                             right=False, labelright=False)
            if a == axc:
                axsi.spines.right.set_position(("axes", 1.25))
                axsi.spines[['right']].set_visible(True)
                axsi.tick_params(axis='y', which='both',
                                 right=True, labelright=True)
                axsi.tick_params(axis='y', labelcolor=c, labelsize=16)
                if n == 1:
                    axsi.set_ylabel('Sea ice concentration ($\%$)',
                                    color=c,
                                    fontsize=16)
                else:
                    axsi.set_ylabel('')

        axa.tick_params(axis='y', which='both', labelsize=16)
        axb.tick_params(axis='y', which='both', left=False, labelleft=False,
                        right=False, labelright=False)
        axc.tick_params(axis='y', which='both', left=False, labelleft=False,
                        right=False, labelright=False)

        cbar = plt.colorbar(p3, orientation="vertical",
                            format=ticker.FormatStrFormatter('%.2f'))
        cbar.set_label(units[var], rotation=90, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.locator = ticker.MaxNLocator(nbins=7)
        cbar.update_ticks()

        axa.set_yticks(depths[var])
        axb.set_yticks(depths[var])
        axc.set_yticks(depths[var])

        axa.grid(True, c='white', lw=0.5, alpha=0.5, zorder=100)
        axb.grid(True, c='white', lw=0.5, alpha=0.5, zorder=100)
        axc.grid(True, c='white', lw=0.5, alpha=0.5, zorder=100)

        # vlines
        axb.vlines(datetime(2021, 9, 6), -220, -50,
                   colors=(33/256, 76/256, 112/256))

        # patches
        rect1 = ptcs.Rectangle(
            (datetime(2021, 9, 3, 0), -220),
            timedelta(days=2),
            170,
            fc="grey",
            ec='grey',
            alpha=0.3
        )
        rect2 = ptcs.Rectangle(
            (datetime(2021, 9, 15, 0), -220),
            timedelta(days=2),
            170,
            fc="grey",
            ec='grey',
            alpha=0.3
        )
        axb.add_patch(rect1)
        axb.add_patch(rect2)

        # Annotation
        axa.text(
            0.035,
            0.095,
            titles[var],
            va='bottom',
            ha='left',
            transform=axs[n].transAxes,
            fontsize=30
        )

        if n == 1:
            axa.set_ylabel('Depth ($m$)', fontsize=16)

    plt.subplots_adjust(
        wspace=0.025,
        hspace=0.1,
        bottom=0.1,
        right=0.85)

    fp = ('Figures/hovmollers/EGU_mooring_hovm_' +
          str(start_date.year)+str(start_date.month).zfill(2) +
          str(start_date.day).zfill(2)+'-'+str(end_date.year) +
          str(end_date.month).zfill(2)+str(end_date.day).zfill(2) +
          '_WOA_corrected_CELLO.svg')
    plt.tight_layout()
    plt.savefig(fp, dpi=900, bbox_inches='tight', transparent=True)


def mooring_TS(ds, start_date, end_date):
    """Plotting a TS diagram of the mooring data.
    Ultimately hoping to see if there is evidence of a front or not, and 
    also if there are interesting non-linear effects."""

    # Basic figure stuff
    plt.rcParams["font.family"] = "serif" # change the base font
    layout = [['ax1','ax2'],
              ['ax3','.'  ]]
    fig, axd = plt.subplot_mosaic(layout,figsize=(7, 7))
    ax1, ax2, ax3 = axd['ax1'], axd['ax2'], axd['ax3']
    
    ds = ds.sel(time=slice(np.datetime64(start_date), np.datetime64(end_date)))

    def plot_TS(ax,ds,d):
        SA_min, SA_max = ds['SA'].sel(depth=d).min().values-0.05, ds['SA'].sel(depth=d).max().values+0.05
        T_min, T_max = ds['T'].sel(depth=d).min().values-0.05, ds['T'].sel(depth=d).max().values+0.05
        SA_1D = np.linspace(SA_min,SA_max,50)
        T_1D = np.linspace(T_min,T_max,50)
        rho_2D = np.zeros((50,50))
        p = ds['p_from_z'].sel(depth=d).mean().values # For now we're just going to look at one depth and hence one pressure
        for col,s in enumerate(SA_1D):
            for row,t in enumerate(T_1D):
                rho_2D[row,col] = gsw.rho_t_exact(s,t,p) - 1000
        print(rho_2D)

        CS = ax.contour(SA_1D,T_1D,rho_2D,colors='k')
        ax.clabel(CS, fontsize=9)

        colours = lambda cm,ds : plt.get_cmap(cm)(np.linspace(0, 1, len(ds['time'])))
        sc = ax.scatter(ds['SA'].sel(depth=d),ds['T'].sel(depth=d),c=colours('plasma',ds),s=0.1)

        ax.set_ylabel("In situ temperature ($℃$)", fontsize=9)
        ax.set_xlabel("Absolute salinity ($g$ $kg^{-1}$)", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

        return sc, colours

    sc,colours=plot_TS(ax1,ds,-50)
    sc,colours=plot_TS(ax2,ds,-125)
    sc,colours=plot_TS(ax3,ds,-220)
    
    ax1.set_title("50 m depth", fontsize=11)
    ax2.set_title("125 m depth", fontsize=11)
    ax3.set_title("220 m depth", fontsize=11)
    plt.suptitle("Mooring sensor TS diagrams")

    # The following is almost directly from Copilot, and it handles the colourbar
    dates = [pd.Timestamp(date).to_pydatetime() for date in ds['time'].values]
    norm = plt.Normalize(dates[0].toordinal(), dates[-1].toordinal()) 
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm) 
    cbar_ax = fig.add_axes([0.6, 0.1, 0.025, 0.35])
    cbar = plt.colorbar(sm, ax=ax3,cax=cbar_ax)
    tick_locs = np.linspace(dates[0].toordinal(), dates[-1].toordinal(), 10)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([pd.Timestamp.fromordinal(int(tick)).strftime('%Y-%m-%d') for tick in tick_locs])
    #cbar.set_label('Date', fontsize=9)

    plt.tight_layout()
    plt.savefig('TS.png',dpi=1200)


def contents_CELLO(ds):
    """Similar idea to the TS diagrams, but with temp and salt time series
    Ultimately hoping to see if there is evidence of a front or not, and
    also if there are interesting non-linear effects."""

    start_date = datetime(2021, 9, 4)
    end_date = datetime(2021, 9, 18)
    window = 24
    dt = True

    # Basic figure stuff
    plt.rcParams["font.family"] = "serif"

    layout = [['ax1', 'ax1', 'ax1', 'ax1', 'ax1', '.'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax1', '.'],
              ['ax1', 'ax1', 'ax1', 'ax1', 'ax1', '.'],
              ['ax2', 'ax2', 'ax2', 'ax2', 'ax2', '.'],
              ['ax2', 'ax2', 'ax2', 'ax2', 'ax2', '.'],
              ['ax2', 'ax2', 'ax2', 'ax2', 'ax2', '.'],
              ['.', '.', '.', '.', '.', '.']]
    fig, axd = plt.subplot_mosaic(layout, figsize=(12, 5), sharex=True)
    ax1, ax2 = axd['ax1'], axd['ax2']

    ds = ds.sel(time=slice(
        np.datetime64(start_date-timedelta(days=3)),
        np.datetime64(end_date+timedelta(days=3))
    ))

    def plotter(var, d, c, ax):
        da = ds[var].sel(depth=d).rolling(time=window, center=True).mean()
        if dt is True:
            # Divide by the delta t (2 hours, or 7200 seconds)
            # multiply by seconds-per-day (86400) to get the per-day-ROC
            da = da.diff(dim='time')/(7200/86400)
            # Take the rolling mean again (a bit unscientific)
            da = da.rolling(time=window, center=True).mean()
        p, = da.plot(ax=ax, c=c, label=str(d)+' m')
        if dt is True:
            ylim = abs(da).max(skipna=True)
            ax.set_ylim(-(ylim+0.05*ylim), (ylim+0.05*ylim))
            ax.hlines(0, da.time.isel(time=0), da.time.isel(time=-1),
                      colors='k')
            ax.vlines(datetime(2021, 9, 7, 6, 0, 0),
                      -(ylim+0.05*ylim),
                      (ylim+0.05*ylim),
                      colors='k')
            ax.vlines(datetime(2021, 9, 11, 3, 0, 0),
                      -(ylim+0.05*ylim),
                      (ylim+0.05*ylim),
                      colors='k')
            ax.set_xlim(start_date, end_date)
        if dt is False:
            start_slice_aug = datetime(2021, 8, 1, 0, 0, 0)
            end_slice_aug = datetime(2021, 9, 1, 0, 0, 0)
            start_slice_dec = datetime(2021, 12, 5, 0, 0, 0)
            end_slice_dec = datetime(2022, 1, 5, 0, 0, 0)
            ax.set_xlim(
                start_slice_aug-timedelta(days=20),
                end_slice_dec+timedelta(days=20))
            start_mean = da.sel(
                time=slice(np.datetime64(start_slice_aug),
                           np.datetime64(end_slice_aug))).mean().values
            end_mean = da.sel(
                time=slice(np.datetime64(start_slice_dec),
                           np.datetime64(end_slice_dec))).mean().values
            ax.text(start_slice_aug, start_mean, str(start_mean)[0:5],
                    color=c, fontsize=9, horizontalalignment='right',
                    verticalalignment='center')
            ax.text(end_slice_dec, end_mean, str(end_mean)[0:5],
                    color=c, fontsize=9, horizontalalignment='left',
                    verticalalignment='center')
            ax.hlines(start_mean, start_slice_aug, end_slice_aug, colors=c)
            ax.hlines(end_mean, start_slice_dec, end_slice_dec, colors=c)
            ymax, ymin = da.max(skipna=True), da.min(skipna=True)
            ax.set_ylim(ymin, ymax)
            ax.vlines(datetime(2021, 9, 7, 12, 0, 0),
                      ymin-0.05*ymin,
                      ymax+0.05*ymax,
                      colors='k')
            ax.vlines(datetime(2021, 9, 12, 8, 0, 0),
                      ymin-0.05*ymin,
                      ymax+0.05*ymax,
                      colors='k')
        ax.tick_params(axis='y', colors=c)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', labelbottom=False)
        return p

    p1 = plotter('T', -50, 'b', ax1)
    ax1a = ax1.twinx()
    p1a = plotter('T', -125, 'r', ax1a)
    ax1b = ax1.twinx()
    p1b = plotter('T', -220, 'g', ax1b)
    ax1b.spines['right'].set_position(("axes", 1.18))

    p2 = plotter('SA', -50, 'b', ax2)
    ax2a = ax2.twinx()
    p2a = plotter('SA', -125, 'r', ax2a)
    ax2b = ax2.twinx()
    p2b = plotter('SA', -220, 'g', ax2b)
    ax2b.spines['right'].set_position(("axes", 1.18))

    if dt is False:
        ax1_title = "In situ temperature ($℃$)"
        ax2_title = "Absolute salinity ($g$ $kg^{-1}$)"
        sup_title = "Mooring temperature and salintity time series"
        file_name = 'contents_window'+str(window)+'.png'
    else:
        ax1_title = "In situ temperature rate of change ($℃$ $day^{-1}$)"
        ax2_title = ("Absolute salinity rate of change ($g$ $kg^{-1}$"
                     "$day^{-1}$)")
        sup_title = ("Mooring temperature and salintity rates of change"
                     "time series")
        file_name = 'contents_dt_window'+str(window)+'_CELLO.svg'
    ax1.set_title(ax1_title, fontsize=16)
    ax1.tick_params(axis='y', which='both', labelsize=16)
    ax1a.tick_params(axis='y', which='both', labelsize=16)
    ax1b.tick_params(axis='y', which='both', labelsize=16)
    ax2.set_title(ax2_title, fontsize=16)
    plt.suptitle(sup_title, fontsize=30)
    ax2.tick_params(axis='x', labelbottom=True, labelsize=16)
    ax2.tick_params(axis='y', which='both', labelsize=16)
    ax2a.tick_params(axis='y', which='both', labelsize=16)
    ax2b.tick_params(axis='y', which='both', labelsize=16)
    fig.subplots_adjust(wspace=0, hspace=1)
    ax2.legend(handles=[p2, p2a, p2b], loc='upper center',
               bbox_to_anchor=(0.15, -0.15),
               title="Nominal sensor depth",
               fontsize=9,
               title_fontsize=9)
    ax1.xaxis.grid(True)
    ax2.xaxis.grid(True)

    plt.savefig(file_name, dpi=1200, transparent=True)


def compare_CTD_cast_and_mooring(ds_mooring, ds_CTD):
    """Plot two column figure of T and S from start and end of mooring
    time series versus the launch and pickup cruise CTD casts."""

    fig, [ax1, ax2] = plt.subplots(ncols=2,nrows=1)

    # Getting the dates used in the plots
    mooring_start_id, mooring_end_id = 0,-2 #15, -15
    mooring_start_time = ds_mooring['time'].isel(time=mooring_start_id).values
    mooring_end_time = ds_mooring['time'].isel(time=mooring_end_id).values
    CTD_start_time = ds_CTD['datetime'].isel(datetime=0).values
    CTD_end_time = ds_CTD['datetime'].isel(datetime=-1).values   

    # Plotting the start mooring points
    mts = ax1.scatter(ds_mooring['T'].isel(time=15),ds_mooring['p_from_z'],
                      c='k',label='Mooring temperature start')
    mss = ax2.scatter(ds_mooring['S'].isel(time=15),ds_mooring['p_from_z'],
                      c='k',label='Mooring salinity start')

    # Plotting the end mooring points
    mte = ax1.scatter(ds_mooring['T'].isel(time=-15),ds_mooring['p_from_z'],
                      c='r',label='Mooring temperature end')
    mse = ax2.scatter(ds_mooring['S'].isel(time=-15),ds_mooring['p_from_z'],
                      c='r',label='Mooring salinity end')

    # Plotting the first CTD cast
    ctdts, = ax1.plot(ds_CTD['T'].isel(datetime=0),ds_CTD['P'],
                      c='k',label='CTD temperature start')
    ctdss, = ax2.plot(ds_CTD['S'].isel(datetime=0),ds_CTD['P'],
                      c='k',label='CTD salinity start')

    # Plotting the second CTD cast
    ctdte, = ax1.plot(ds_CTD['T'].isel(datetime=-1),ds_CTD['P'],
                      c='r',label='CTD temperature end')
    ctdse, = ax2.plot(ds_CTD['S'].isel(datetime=-1),ds_CTD['P'],
                      c='r',label="CTD salinity end")

    ax1.invert_yaxis()
    ax2.invert_yaxis()

    leg1 = ax1.legend(
        [mts,mte],
        [str(mooring_start_time)[:10],
         str(mooring_end_time)[:10]],
        title="Mooring\n(resampled daily)",
        fontsize='small',
        framealpha=0,
        title_fontsize='small',
        loc="lower left",
    )

    leg2 = ax1.legend(
        [ctdss,ctdse],
        ['PS124 ('+str(CTD_start_time)[:10]+')',
         'PS129: ('+str(CTD_end_time)[:10]+')'],
        title="CTD casts\n(launch and pick up)",
        fontsize='small',
        framealpha=0,
        title_fontsize='small',
        loc="center left",
        bbox_to_anchor = [0, 0.3]
    )

    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"

    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    plt.suptitle("Mooring instruments vs CTD casts")
    ax1.set_title("Temperature")
    ax2.set_title("Salinity")
    ax1.set_xlabel("In situ temperature ($℃$)")
    ax1.set_ylabel("Pressure ($dbar$)")
    ax2.set_xlabel("Practical salinity ($PSU$)")

    plt.savefig('mooring_vs_CTD.png',dpi=600)


if __name__ == "__main__":
    ds = ma.open_mooring_data()
    ds.correct_mooring_salinities()
    ds.append_gsw_vars()
    # plt_hovm_CELLO(ds)
    contents_CELLO(ds)
