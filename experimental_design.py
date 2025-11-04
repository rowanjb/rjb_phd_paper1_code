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
#    prescribed and not do not feedback; equilibrium is not reached;
# 3) Turbulence (for now) is assumed anisotropic and we therefore use
#    the viscosities from the tried-and-true Smagorinsky 2D param,
#    although this should be tested when doing convergence tests;
# 4) Domain needs to just be large enough that instabilities from the
#    plume don't interact too much and don't feel bottom effects by
#    the end of the simulation (could also go a sort of convergence
#    test analogue);
# 5) Resolution should be tested for congergence, but generally at
#    2 m we're in the ballpark of the Taylor microscale (I think...);
# 6) Time scale should capture the plume reaching maximum depth and
#    subsequently rebounding (but again equilibrium is not reached);
# 7) Ice opening/lead size is based on Muchow et al. (2021), i.e., in
#    the ballpark of 100 m; Perovich and Richter-Menge (2000) also
#    discuss <200 m leads, which has the benefit of reduce fetch
#    considerations;
# 8) Forcing time and forcing magnitude: See below;
# 10) Freezing point: Assume /some/ supercooling, e.g., -1.9, following
#    Alex's Supercooled Southern Ocean Waters paper;
# 11) Equation of state: Non-linear TEOS-10 (necessary for our mooring
#    data, I think, as without it we get marginal stability where it
#    does not actually exist (test this before claiming it though),
#    and also it's just more accurate);
# 12) Initial conditions: Stratification for some simulations is based
#    on our mooring observations, taken as an over the days preceding
#    the plume (i.e., preconditioning is already completed), and we
#    follow Martinson (1990) by assuming that the mixed layer
#    is homogenised down to a permanent, thin pycnocline
#    (an important corollary is that we assumed mixing in the ML is
#    responsible for its homogenisation but we do not attempt to
#    quantify its effects on mixing the plume-released thermocline
#    heat upwards); additional simulations with different thermocline/
#    pycnocline slopes would also be a good idea
# 13) Restoring sponges: Not really necessary for short-term simulations


# == Forcing time and forcing magnitude:
# Among the least-well constrained consideration are the forcing time
# and magnitude, i.e., (1) how strong should the ocean-to-atmosphere heat
# flux be, (2) how much salt should be rejected into the upper layer of the
# model, and (3) how should these forcings vary over time. Heat flux (1) is
# simpler than salinity; if we limit the water to some freezing point,
# then MITgcm will only impose heat flux until this temperature is
# reached, regardless of the flux in the forcing file (it is also less
# important in terms of the N2, since we're in an alpha ocean). The
# salt flux (2) is more difficult, since it is related to the rate of ice
# growth and the percentage salt distillation by mass. Some literature
# is useful here, including work by Perovich and Wadhams. E.g.,
#  - Perovich and Richter-Menge (2000):
#     - 15--20 cm growth in first few days of lead opening
#     - Bulk salinity of the resultant ice ranges from 7.3 --25.1
#  - Wilkinson and Wadhams (2003):
#     - Antarctic pancake salinity: 4 psu after 4 days
#     - Alternatively modelled (frazil) as 18.3exp(-1.2/2)+1.5=11.5 PSU
#  - (I read somewhere that lab values can reach 10 cm / day but I can't
#     remember where...)
#  - I also like the practicality of my old MUN Sea Ice Eng course notes
#    from Claude Daley, but I would hesitate to rely on this in a paper;
#    the relevant portion is ultimately based on Stefan's law (Stefan, 1891)
# Consider a conservative assumed 10 cm of growth in one day,
# with a drop in PSU from ~35 to ~10; this gives us
# (1m)*(1m)*(0.1m/day)*(25kg/m3) = 2.5kg/day = 0.0289g/s (per m2).
# Alternatively, Daley (Fig. 3.16, Eq. 8) offers an equation based on 
# Stefan (1891), which is in German. Leppäranta (1993) gives an updated
# overview of Stefan (1891), where:
# ice thickness = sqrt(3.3^2 [sqrt(cm/C/d)] * FDD)
# With a temperature difference of ~-20 (based on ERA5) this gives
# sqrt(3.3^2*20) = 14.76 cm, which they note is an ideal/upper bound on
# what might actually be expected. Daley discusses an updated "practical"
# formulation from Cammaert and Muggeridge (1988): 0.025*sqrt(FDD)=11.18 cm.
# Since these values are all roughly aligned, I will use Stefan's Law
# (although I really wish I could read it) because (3) it gives a smooth
# salt rejection. Note that I will still use maximum cooling, because
# as noted heat flux is regardless limited by the freezing point.


def initial_profiles(ds, Nx, Ny, Nr, dr):
    """Create initial T and S profiles based on the mooring.
    Parameters:
        ds: input dataset of mooring observations
        Nx, Ny, Nr: Grid points in each dimension
        dr: Constant vertical spacing
    Returns:
        Saves input binary for MITgcm"""

    # Consider that we only have 3 data points: 1 in the ML, 1 beneath
    # the pycnocline, and 1 either just above, within, or just below the
    # pycnocline. The basic stratification is to have a homogenised ML,
    # a 'cline with two slopes, and then a homogenised sub-surface. (If
    # we say that there is only 1 slope, our constraints from the mooring
    # end up yielding instabilities)

    ds = ds.sel(depth=[-50, -125, -220])
    ds['sigma0'] = gsw.sigma0(ds['SA'], ds['CT'])
    ds = ds.sel(time=slice(dt(2021, 9, 12, 12), dt(2021, 9, 13, 12)))
    ds = ds.mean('time')

    def linear_profiles(da, depth):
        '''Returns an interpolated (linear) salinity, temperature, or
        depth. Depth is negative.'''

        # If assumed homogenous (i.e., above -50 m or below -220)
        if depth > -50:
            return da.sel(depth=-50)
        elif depth < -220:
            return da.sel(depth=-220)

        # If between -50 and -220, calculate the gradient
        # Also define linear function of form depth p_grad + b = sigma0
        if depth <= -50 and depth >= -125:
            gradient = (da.sel(depth=-50) - da.sel(depth=-125))/75
            b = da.sel(depth=-125) - (-125*gradient)
        elif depth < -125 and depth >= -220:
            gradient = (da.sel(depth=-125) - da.sel(depth=-220))/95
            b = da.sel(depth=-220) - (-220*gradient)
        else:
            print("Something's wrong!")
            exit()

        # Now that we know the value of b, we can query the expression
        p = b + (depth*gradient)

        return p.values

    # Calculate the profiles
    depths = np.arange((-1)*dr, (-1)*(Nr+1)*dr, (-1)*dr)
    SAs, pts, sigma0s = [], [], []
    for depth in depths:
        SA = linear_profiles(ds['SA'], depth)
        pt = linear_profiles(ds['pt'], depth)
        CT = gsw.CT_from_pt(SA, pt)
        sigma0 = gsw.sigma0(SA, CT)
        SAs.append(SA)
        pts.append(pt)
        sigma0s.append(sigma0)

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(ds['SA'], ds['depth'], c='black')
    ax2 = ax.twiny()
    ax2.scatter(ds['pt'], ds['depth'], c='red', marker='x')
    ax3 = ax.twiny()
    ax3.scatter(ds['sigma0'], ds['depth'], c='blue', marker='.')
    ax3.plot(sigma0s, depths, c='blue')
    ax.plot(SAs, depths, c='black')
    ax2.plot(pts, depths, c='red')
    plt.savefig('tmp_figs/initial_conditions.png', dpi=1200)

    # Save binaries
    # Docs say order is Nx Ny Nr, but this works with order='F', and
    # I've tested it against binaries that came in the verification runs
    xmitgcm.utils.write_to_binary(
        np.tile(pts, (Nx, Ny, 1)).flatten(order='F'),
        'bin_init_pt_'+str(Nx)+'x'+str(Ny)+'x'+str(Nr)+'.bin')
    xmitgcm.utils.write_to_binary(
        np.tile(SAs, (Nx, Ny, 1)).flatten(order='F'),
        'bin_init_SA_'+str(Nx)+'x'+str(Ny)+'x'+str(Nr)+'.bin')


def initial_velocities(Nx, Ny, Nr, dr):
    """Initialise at zero; might need to reconsider the order of axes
    and the order='C' line, but with all zeros it doesn't matter."""
    U = np.full((Nr, Nx, Ny), 0)
    V = np.full((Nr, Nx, Ny), 0)
    xmitgcm.utils.write_to_binary(
        U.flatten(order='C'),
        'bin_init_U_'+str(Nx)+'x'+str(Ny)+'x'+str(Nr)+'.bin')
    xmitgcm.utils.write_to_binary(
        V.flatten(order='C'),
        'bin_init_V_'+str(Nx)+'x'+str(Ny)+'x'+str(Nr)+'.bin')


def initial_eta(Nx, Ny):
    """Initialise at zero."""
    Eta = np.full((Nx, Ny), 0)
    xmitgcm.utils.write_to_binary(
        Eta.flatten(order='C'),
        'bin_init_Eta_'+str(Nx)+'x'+str(Ny)+'.bin')


def forcing_Q(Nx, Ny, dx, lead_width):
    """For now we assume 1 day of forcing followed by the
    reorganization of layers. We let the forcing be some high value
    like 200 W/m2 (this is supported in the literature), because
    it doesn't strictly matter since we're limited to the freezing
    point of sea water anyway.
    Parameters:
        Nx, Ny: Number of grid points
        dx: Grid spacing (m), assumed to be square cells
        lead_width: Lead width in m (assumed parallel opening)
            *lead width will get rounded to the nearest cell
    Returns:
        Saves binary for MITgcm"""

    Q = np.full(36, 200)
    Q[25:] = 0

    # Now we can save this as a 3D binary
    # Note we init with 0.0 because we want floats not ints
    Qf = np.full((36, Nx, Ny), 0.0)  # Array of salt flux rates
    lead_cells = int(np.floor(lead_width/dx))  # Width of the lead in cells
    real_width = int(lead_cells*dx)  # Useful when saving, maybe
    left_coord = int(np.ceil((Ny-lead_cells)/2))
    right_coord = left_coord+lead_cells
    for n, heat_flux in enumerate(Q):
        Qf[n, :, left_coord:right_coord] = heat_flux
    # Note the order of axes and flattening; this setup seems to work
    xmitgcm.utils.write_to_binary(
        Qf.flatten(order='C'),
        'bin_forc_Q_36x'+str(Nx)+'x'+str(Ny)+'_'+str(real_width)+'m_lead.bin')


def forcing_salt(Nx, Ny, dx, lead_width):
    """We base our ice growth rate on Stefan's Law and use the
    coefficient (0.025) suggested by Cammaert and Muggeridge (1988).
    Leppäranta (1993) provides a good overview of Stefan's (1891)
    idealised formulation. With an air temperature of -15.15 C (ERA5
    on Sep 12, 2021) we get 0.09730 m of ice after 1 day. We then cease
    forcing for 12 hours to let the water reorganise itself.
    Parameters:
        Nx, Ny: Number of grid points
        dx: Grid spacing (m), assumed to be square cells
        lead_width: Lead width in m (assumed parallel opening)
            *lead width will get rounded to the nearest cell
    Returns:
        Saves binary for MITgcm"""

    hours = np.arange(37)  # 36 hours plus 1 more for differentiation
    ice_thickness = [0.037*np.sqrt(i*17.15/24) for i in hours]
    ice_growth = np.diff(ice_thickness)

    # But we only want one day of growth
    ice_growth[25:] = 0

    # Note if we assume 30 PSU are rejected, then we can calculate flux
    # 1m x 1m x [growth rate] m/hr x 30 kg/m3 = [salt flux] kg/hr
    salt_rejection = ice_growth*30/3600*1000

    # Plot to show what I mean
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(hours[:-1], np.cumsum(ice_growth), c='k')
    ax2.plot(hours[:-1], salt_rejection, c='b')
    ax.set_ylabel("Ice thickness (black line)")
    ax2.set_ylabel("Salt rejection (blue line)")
    plt.savefig("tmp_figs/plume_sflux.png", dpi=1200)

    # Now we can save this as a 3D binary
    # Note we init with 0.0 because we want floats not ints
    sf = np.full((36, Nx, Ny), 0.0)  # Array of salt flux rates
    lead_cells = int(np.floor(lead_width/dx))  # Width of the lead in cells
    real_width = int(lead_cells*dx)  # Useful when saving, maybe
    left_coord = int(np.ceil((Ny-lead_cells)/2))
    right_coord = left_coord+lead_cells
    for n, salt_rejection_rate in enumerate(salt_rejection):
        sf[n, :, left_coord:right_coord] = (-1)*salt_rejection_rate
    # Note the order of axes and flattening; this setup seems to work
    xmitgcm.utils.write_to_binary(
        sf.flatten(order='C'),
        'bin_forc_SA_36x'+str(Nx)+'x'+str(Ny)+'_'+str(real_width)+'m_lead.bin')


def plot_initial_conditions(Nx, Ny, Nr, dr, lead_width, i):
    """Creates a non-paper quality figure showing the binaries."""

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(14, 14))

    # Q forcing
    fp = "bin_forc_Q_36x"+str(Nx)+"x"+str(Ny)+"_"+str(lead_width)+"m_lead.bin"
    Q = xmitgcm.utils.read_raw_data(
        fp, shape=(36, Nx, Ny), dtype=np.dtype('>f4'))
    P = ax[0][0].pcolormesh(Q[0, :, :])
    ax[0][0].set_title("TFLUX at $t=0$ ($W$)")
    ax[0][0].set_xlabel("X cells")
    ax[0][0].set_ylabel("Y cells")
    fig.colorbar(P, ax=ax[0][0])

    # S forcing
    fp = "bin_forc_SA_36x"+str(Nx)+"x"+str(Ny)+"_"+str(lead_width)+"m_lead.bin"
    Sf = xmitgcm.utils.read_raw_data(
        fp, shape=(36, Nx, Ny), dtype=np.dtype('>f4'))
    P = ax[0][1].pcolormesh(Sf[0, :, :])
    ax[0][1].set_title("SFLUX at $t=0$ ($g/m^{2}/s$)")
    ax[0][1].set_xlabel("X cells")
    ax[0][1].set_ylabel("Y cells")
    fig.colorbar(P, ax=ax[0][1])

    # Forcing time series
    ax[0][2].plot(Q[:, int(Nx/2), int(Ny/2)], c='r')
    ax[0][2].set_ylabel("TFLUX in lead ($W$) (red)")
    ax[0][2].set_xlabel("Time ($h$)")
    ax[0][2].set_title("Buoy forcing over time\n(within the lead)")
    axb = ax[0][2].twinx()
    axb.plot(Sf[:, int(Nx/2), int(Ny/2)], c='b')
    axb.set_ylabel("SFLUX in lead ($g/m^{2}/s$) (blue)")

    # SSH
    fp = "bin_init_Eta_"+str(Nx)+"x"+str(Ny)+".bin"
    Eta = xmitgcm.utils.read_raw_data(
          fp, shape=(Nx, Ny), dtype=np.dtype('>f4'))
    P = ax[1][0].pcolormesh(Eta[:, :])
    ax[1][0].set_title("SSH at $t=0$ ($m$)")
    ax[1][0].set_xlabel("X cells")
    ax[1][0].set_ylabel("Y cells")
    fig.colorbar(P, ax=ax[1][0])

    # U vel
    fp = "bin_init_U_"+str(Nx)+"x"+str(Ny)+"x"+str(Nr)+".bin"
    U = xmitgcm.utils.read_raw_data(
        fp, shape=(Nx, Ny, Nr), dtype=np.dtype('>f4'))
    P = ax[1][1].pcolormesh(U[:, int(Ny/2), :])
    ax[1][1].set_title("U velocity at $t=0$ ($m/s$)")
    ax[1][1].set_xlabel("Y cells")
    ax[1][1].set_ylabel("Z cells")
    fig.colorbar(P, ax=ax[1][1])

    # V vel
    fp = "bin_init_V_"+str(Nx)+"x"+str(Ny)+"x"+str(Nr)+".bin"
    V = xmitgcm.utils.read_raw_data(
        fp, shape=(Nx, Ny, Nr), dtype=np.dtype('>f4'))
    P = ax[1][2].pcolormesh(V[:, int(Ny/2), :])
    ax[1][2].set_title("V velocity at $t=0$ ($m/s$)")
    ax[1][2].set_xlabel("Y cells")
    ax[1][2].set_ylabel("Z cells")
    fig.colorbar(P, ax=ax[1][2])

    # Potential temperature
    fp = "bin_init_pt_"+str(Nx)+"x"+str(Ny)+"x"+str(Nr)+".bin"
    pt = xmitgcm.utils.read_raw_data(
         fp, shape=(Nr, Nx, Ny), dtype=np.dtype('>f4'))
    P = ax[2][0].pcolormesh(pt[:, :, int(Ny/2)])
    ax[2][0].set_title("Potential temp. at $t=0$ ($℃$)")
    ax[2][0].set_xlabel("Y cells")
    ax[2][0].set_ylabel("Z cells")
    fig.colorbar(P, ax=ax[2][0])

    # Absolute salinity
    fp = "bin_init_SA_"+str(Nx)+"x"+str(Ny)+"x"+str(Nr)+".bin"
    SA = xmitgcm.utils.read_raw_data(
         fp, shape=(Nr, Nx, Ny), dtype=np.dtype('>f4'))
    P = ax[2][1].pcolormesh(SA[:, :, int(Ny/2)])
    ax[2][1].set_title("Absolute salinity at $t=0$")
    ax[2][1].set_xlabel("Y cells")
    ax[2][1].set_ylabel("Z cells")
    fig.colorbar(P, ax=ax[2][1])

    # Initial profiles
    zcoord = np.linspace(int((-1)*dr/2), (-1)*Nr*dr-int((-1)*dr/2), Nr)
    print(np.shape(zcoord))
    print(np.shape(pt))
    ax[2][2].plot(pt[:, int(Nx/2), int(Ny/2)], zcoord, c='r')
    ax[2][2].set_xlabel("Pot. temp. ($℃$) (red)")
    ax[2][2].set_ylabel("Depth ($m$)")
    ax[2][2].set_title("Initial profiles")
    axb = ax[2][2].twiny()
    axb.plot(SA[:, int(Nx/2), int(Ny/2)], zcoord, c='b')
    axb.set_xlabel("Abs. sal. (blue)\nsigma0 ($kg/m^3$) (black)")
    axc = ax[2][2].twiny()
    CT = gsw.CT_from_pt(SA[:, int(Nx/2), int(Ny/2)],
                        pt[:, int(Nx/2), int(Ny/2)])
    axc.plot(gsw.sigma0(SA[:, int(Nx/2), int(Ny/2)], CT), zcoord)

    plt.subplots_adjust(hspace=0.75, wspace=0.75)
    plt.suptitle("Initial conditions and forcings")
    plt.savefig("binary_overview_"+i+".png")


if __name__ == "__main__":
    Nx, Ny, Nr, dr, lead_width, i = 33, 594, 99, 4, 100, "091"
    ds = mtsa.open_mooring_data()
    ds = mtsa.correct_mooring_salinities(ds)
    ds = mtsa.append_gsw_vars(ds)
    initial_profiles(ds, Nx, Ny, Nr, dr)
    initial_velocities(Nx, Ny, Nr, dr)
    initial_eta(Nx, Ny)
    forcing_Q(Nx, Ny, dr, lead_width)
    forcing_salt(Nx, Ny, dr, lead_width)
    plot_initial_conditions(Nx, Ny, Nr, dr, lead_width, i)
