# File for creating TS diagrams, which qualitatively show mixing
# from the plume

import xmitgcm
import gsw
import matplotlib.pyplot as plt
import numpy as np


def calc_density_lines(ds):
    """Calcs lines of constant density for the TS diagram
    Cite: https://oceanpython.org/2013/02/17/t-s-diagram/"""

    # Figure out boudaries (mins and maxs)
    smin = ds['S'].min().values - (0.001 * ds['S'].min()).values
    smax = ds['S'].max().values + (0.001 * ds['S'].max()).values
    tmin = ds['CT'].min().values - (0.1 * ds['CT'].max()).values
    tmax = ds['CT'].max().values + (0.1 * ds['CT'].max()).values

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax-smin)/0.1 + 1, 0))
    ydim = int(round((tmax-tmin) + 1, 0))

    # Create empty grid of zeros
    dens = np.empty((ydim, xdim))

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(0, ydim-1, ydim) + tmin
    si = np.linspace(0, xdim-1, xdim)*0.1 + smin

    # Loop to fill in grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.sigma0(si[i], ti[j])

    return si, ti, dens


def plot_TS_diagram(fp, dt=4, dx=4):

    # Open data
    ds = xmitgcm.open_mdsdataset(
        fp, prefix=['S', 'T'], delta_t=dt, ref_date="2021-9-13 12:0:0")

    # Cut around plume
    mid_id = int(np.floor(ds['XC'].isel(XC=-1)/(dx*2)))
    ds = ds.isel(XC=slice(int(mid_id-(300/dx)), int(mid_id+(300/dx)), 10))
    ds = ds.isel(YC=slice(0, -1, 5))

    # Calc conservative temp
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])

    # Get density lines
    salt, temp, dens = calc_density_lines(ds)
    fig, ax1 = plt.subplots()
    cm = 1/2.54
    fig.set_figwidth(8*cm)
    fig.set_figheight(8*cm)
    C = ax1.contour(salt, temp, dens)
    plt.clabel(C, fontsize=9, inline=1, fmt='%1.2f')
    ax1.scatter(ds['S'].isel(time=-1), ds['CT'].isel(time=-1), c='r', s=0.1,
                label="$T_{end}$")
    ax1.scatter(ds['S'].isel(time=0), ds['CT'].isel(time=0), c='k', s=0.1,
                label="$T_{start}$")
    ax1.legend(loc='lower right')
    ax1.set_title("Mixing in model", fontsize=12)
    ax1.set_ylabel("Conservative temperature ($℃$)", fontsize=9)
    ax1.set_xlabel("Absolute salinity ($g$ $kg^{-1}$)", fontsize=9)
    ax1.tick_params(axis='both', labelsize=9)
    plt.subplots_adjust(left=0.25, bottom=0.15)
    plt.savefig('figure_model_TS.png', dpi=1200)


if __name__ == "__main__":
    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_099/'
    plot_TS_diagram(fp)
