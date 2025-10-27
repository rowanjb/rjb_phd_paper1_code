# General script for preparing my suite of MITgcm experiments.
# Ultimately, I am here referring to the preparation of binary files
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
    # only the slope of the 'clines if we assume three basic "layers" (
    # the ML, the pycnocline, and the CDW)
    ds = ds.sel(depth=[-50, -125, -220])
    ds['sigma0'] = gsw.sigma0(ds['SA'], ds['CT'])
    ds = ds.sel(time=slice(dt(2021, 9, 8), dt(2021, 9, 9)))

    def three_layer_profiles(da, p_grad, depth):
        '''Describe a piecewise function giving a profile at a specified
        depth. Angle is specified by the vertical gradient, e.g., -0.01.
        Depth is negative.'''

        # Known values
        p1, p2, p3 = da.values

        # Linear function of 'cline, of form depth p_grad + b = sigma0
        b = p2 - (-125*p_grad)
        new_p = b + (depth*p_grad)

        if new_p < p1:
            new_p = p1
        elif new_p > p3:
            new_p = p3

        return new_p

    # Since one question that we're interested in is the effect of the
    # relative T and S gradients, this is what we'll vary. This will
    # cause a ballooning in the pot dens profile.
    da_SA = ds['SA'].mean('time')
    da_pt = ds['pt'].mean('time')
    da_sigma0 = ds['sigma0'].mean('time')

    fig, ax = plt.subplots()
    ax.scatter(da_SA, da_SA['depth'], c='black')
    ax2 = ax.twiny()
    ax2.scatter(da_pt, da_pt['depth'], c='red', marker='x')
    ax3 = ax.twiny()
    ax3.scatter(da_sigma0, da_sigma0['depth'], c='blue', marker='.')

    # N2S: You'll want to possibly test if this holds for these grads or
    # for these ratios
    SA_grads = [-0.003]
    pt_grads = [-0.02]

    depths = np.arange(-2, -396, -2)
    for n, SA_grad in enumerate(SA_grads):
        pt_grad = pt_grads[n]
        new_SAs, new_pts, new_simga0s = [], [], []
        for d in depths:
            new_SA = three_layer_profiles(da_SA, SA_grad, d)
            new_pt = three_layer_profiles(da_pt, pt_grad, d)
            CT = gsw.CT_from_pt(new_SA, new_pt)
            new_SAs.append(new_SA)
            new_pts.append(new_pt)
            new_simga0s.append(gsw.sigma0(new_SA, CT))
        ax3.plot(new_simga0s, depths, label=str(pt_grad), c='blue')
        ax.plot(new_SAs, depths, label=str(SA_grad), c='black')
        ax2.plot(new_pts, depths, label=str(pt_grad), c='red')
    plt.savefig('test.png', dpi=1200)


if __name__ == "__main__":
    #ds = mtsa.open_mooring_data()
    #ds = mtsa.correct_mooring_salinities(ds)
    #ds = mtsa.append_gsw_vars(ds)
    ds = xr.open_dataset('tmp.nc')
    initial_profiles(ds)
