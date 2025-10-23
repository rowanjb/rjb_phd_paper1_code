# General script for preparing my suite of MITgcm experiments.
# Ultimately, I am referring to the preparation of binary files
# for initialising my simulations.

import mooring_time_series_analyses as mtsa
from datetime import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import gsw
import xmitgcm
import xmitgcm.utils

# For creating binaries, I rely on documentation located here:
# https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html
# I use the following assumptions to constrain my list of experiments:
# 1) There is no coupled see ice model, which simplifies the number of
#    variables involved;
# 2) Consequently heat and salinity fluxes through the surface are
#    prescribed and not do not feedback; equilibrium is not reached
# 3) Turbulence (for now) is assumed anisotropic and we therefore use
#    the viscosities from the tried-and-true Smagorinsky 2D param.
# 4) Domain...
# 5) Resolution: At 2 m we're in the ballpark of the Taylor microscale
# 6) Time scale...
# 7) Ice opening/lead size...
# 8) Freezing point...
# 9) Equation of state...
# 10) Initial conditions: Stratification is based on our mooring
#     observations, taken as an over the days preceding the plume
#     (i.e., preconditioning is already completed), and we
#     follow Martinson (1990) by assuming that the mixed layer
#     is homogenised down to a permanent, thin pycnocline
#     (an important corollary is that we assumed mixing in the ML is
#     responsible for its homogenisation but we do not attempt to
#     quantify its effects on mixing the plume-released thermocline
#     heat upwards)
# 11) Restoring sponges: No...


def initial_profiles(ds):
    """Create initial T and S profiles"""

    # Consider that we only have 3 data points: 1 in the ML, 1 beneath
    # the pycnocline, and 1 either just above, within, or just below the
    # pycnocline. We can create a simple suite of profiles by varying
    # only the slope of the 'clines.
    ds = ds.sel(depth=[-50, -125, -220])
    ds['sigma0'] = gsw.sigma0(ds['SA'], ds['CT'])
    ds = ds.sel(time=slice(dt(2021, 9, 12), dt(2021, 9, 13)))

    def density_profiles(da, pd_grad, depth):
        '''Describe a piecewise function giving density at a specified
        depth. Pycnocline angle is specified by the vertical potential
        density gradient, e.g., -0.01. Depth is negative.'''

        # Known potential densities
        pd1, pd2, pd3 = da.values

        # Linear function of pycnocline, of form depth pd_grad + b = sigma0
        b = pd2 - (-125*pd_grad)
        new_pd = b + (depth*pd_grad)

        if new_pd < pd1:
            new_pd = pd1
        elif new_pd > pd3:
            new_pd = pd3

        return new_pd

    da = ds['sigma0'].mean('time')

    fig, ax = plt.subplots()
    ax.scatter(da, da['depth'])

    depths = np.arange(-2, -396, -2)
    for pd_grad in [-0.00025, -0.0005, -0.00075, -0.001, -0.00125, -0.0015]:
        new_pds = []
        for d in depths:
            new_pds.append(density_profiles(da, pd_grad, d))
        ax.plot(new_pds, depths, label=str(pd_grad))
    ax.grid()
    ax.legend()
    plt.savefig('test.png')


if __name__ == "__main__":
    #ds = mtsa.open_mooring_data()
    #ds = mtsa.correct_mooring_salinities(ds)
    #ds = mtsa.append_gsw_vars(ds)
    ds = xr.open_dataset('tmp.nc')
    initial_profiles(ds)
