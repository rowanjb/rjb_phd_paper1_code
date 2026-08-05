# This script is to estimate things like heat entrainment rates (across
# all simulations) ultimately following theories from Martinson, Wilson,
# and others.
# Rowan Brown, July 2026

import f90nml
import re
import numpy as np
import scipy
import xarray as xr
import xmitgcm
import gsw


# Get delta_t
def get_delta_t(fp):
    nml = f90nml.read(fp+"/data")
    return nml["PARM03"]["deltaT"]


# Opening the model dataset
def open_dataset(fp):
    dt = get_delta_t(fp)
    ds = xmitgcm.open_mdsdataset(fp, prefix=['S', 'T'], delta_t=dt)
    ds['Z'] = ds['Z'].astype('<f4')
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['sigma0'] = gsw.sigma0(ds['S'], ds['CT'])
    return ds


# Open the diagnostic output
def open_diagnostics(fp):
    dt = get_delta_t(fp)
    ds = xmitgcm.open_mdsdataset(fp, prefix=['surfDiag'], delta_t=dt)
    ds['Z'] = ds['Z'].astype('<f4')
    return ds


# Calculate the heat content anomaly
# Note the tref doesn't matter much when we calculate an anomaly,
# because it is affected only by the density difference term
def calc_hc(ds, tref=-2):
    ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
    ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
    ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
    ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
    ds['HC'] = ds['rho']*ds['cp']*(ds['CT']-tref)*ds['drF']*ds['rA']
    ds['HC_perlevel'] = ds['HC'].sum(['XC', 'YC'])
    return ds  # Units of HC are J (total per cell)


# Calculte the vertical heat content anomaly
def calc_hc_anom(ds):

    # First calc the hc
    ds = calc_hc(ds)

    # Next define the start and end of forcing
    t1 = np.timedelta64(0, 'h')
    t2 = np.timedelta64(24, 'h')

    # Calculare the HC change
    da = ds['HC_perlevel'].sel(time=t2) - ds['HC_perlevel'].sel(time=t1)
    hc_anom = da/1e12/da['drF']  # Units become TJ per m of depth

    return hc_anom


# Calculate the heat entrainment at a specified depth
def heat_entrainment(runid, depth):

    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'
    ds = open_dataset(fp + runid)
    da = calc_hc_anom(ds)
    da = da*da['drF']  # Convert to heat per layer
    da = da.cumsum()

    return da.interp(Z=depth).to_numpy()


# Calculate the entrainment depth for a specified simulation
# and also the thermal barrier
def ze_tb(runid):

    # First we need some data from the namelist data file
    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'
    data = f90nml.read(fp+runid+"/data")
    sf_file = data["PARM05"]["saltFluxFile"]
    eFP = data["PARM03"]["externForcingPeriod"]

    # We calculate the salt flux per m^2 in the lead
    sf_bin = fp + runid + '/' + sf_file
    match = re.search(r"(\d+)x(\d+)x(\d+)", sf_bin)  # From googla ai
    integration_time, Nx, Ny = map(int, match.groups())
    sf = xmitgcm.utils.read_raw_data(
        sf_bin, shape=(integration_time, Nx, Ny), dtype=np.dtype('>f4'))
    sf = sf[:, int(Nx/2), int(Ny/2)]*(-1)
    sf = sf[0:40]  # Cut it arbitrarily after 40 records
    time = [eFP*i for i in np.arange(len(sf))]
    total_salt_flux = scipy.integrate.trapezoid(sf, time)  # In g/m2
    total_salt_flux = total_salt_flux/1000  # We need it in kg

    # Then we extract the initial sa profile (XC and YC are arbitrary)
    # and calculate the sa anomaly in a loop
    ds = open_dataset(fp+runid)
    sa = ds["S"].isel(time=0, YC=5, XC=5)
    sa_anom_at_each_depth = []
    for zn, z in enumerate(sa['Z']):  # Basically here we're calculating
        sn = sa.isel(Z=zn).to_numpy()  # the salt anomaly to entrain
        sa_anom = sn-sa                # all the water down to each cell,
        sa_anom = sa_anom.where(sa_anom >= 0)  # then we can interp to
        sa_anom_at_each_depth.append(  # find the z_e associated with our
            (sa_anom * sa_anom['drF']).sum('Z').to_numpy())  # salt anom.
    sa = sa.to_dataset(name="sa")  # This is simply so that we can easily
    sa["sa_anom"] = xr.DataArray(  # add z coords to the sa_anom data,
        np.asarray(sa_anom_at_each_depth),  # i.e., so that we can interp
        dims=["Z"],                         # later and get z_e
        coords={"Z": sa.Z},
    )

    # Finally we can interpolate and get the z_e entrainment depth
    # associated with our salt anomaly/rejection.
    # Note the sa_anom values are not always unique (i.e., within the
    # mixed layer and below 220 m) but this should still work because
    # our salt rejection amount is guaranteed to be in the halocline.
    z_interp = np.interp(
        total_salt_flux,
        sa["sa_anom"].values,
        sa["sa_anom"].Z.values
    )

    # Now that we have the entrainment depth ze, we can calculate the
    # heat that would be released from the thermocline.
    # Note the tref here is just the ML temperature (in CT), and we
    # pass in time=0 because we only care about the initial conditions.
    # XC, YC, and similar choices are arbitrary
    ds = calc_hc(ds.isel(time=0), tref=ds["CT"].isel(XC=5, YC=5, time=0, Z=6))
    ds = ds.isel(XC=5, YC=5)  # Select an arbitrary column
    da = ds['HC']/1e6/ds['rA']  # Now in MJ/m2
    tb = da.cumsum().interp(Z=z_interp).to_numpy()

    # We can also calculate the total (domain-wide) heat entrainment based
    # on the thermal barrier and the lead area; we can get the lead area
    # via the diagnostics
    diag = open_diagnostics(fp+runid).isel(time=0)  # Only need one time
    diag = diag.reset_coords("rA")  # Want area to be a var, not a coord
    diag = diag.where(diag['SFLUX'] > 0)
    total_tb_entrainment = (tb*diag['rA']).sum(['XC', 'YC']).to_numpy()
    total_tb_entrainment = total_tb_entrainment/1e6  # MJ to TJ

    # For comparison, we can also calculate the modelled heat flux at ze
    # Note different z_interp values can drastically change the results
    real_flux = heat_entrainment(runid, z_interp)

    # Print messages
    print("================================")
    print("Run: "+runid)
    print("Salt rejection: "+str(total_salt_flux)[0:7]+" kg/m2")
    print("Entrainment depth (ze): "+str(z_interp)[0:8]+" m")
    print("Thermal barrier (tb): "+str(tb)[0:8]+" MJ/m2")
    print("Domain-wide tb entrainment: "+str(total_tb_entrainment)[:5]+" TJ")
    print("Modelled flux through ze: "+str(real_flux)[:6]+" TJ")


# Mixed layer function, developed using ChatGPT
def mixed_layer(ds, method="gradient", threshold=0.03, ref_depth=10):

    rho = ds["sigma0"]

    if method == "gradient":
        drhodz = rho.differentiate("Z")
        ds['mld'] = -drhodz.idxmax("Z")
    elif method == "threshold":
        rho_ref = (rho.sel(Z=-ref_depth, method="nearest"))
        delta_rho = rho - rho_ref  # density anomaly relative to reference
        mask = delta_rho >= threshold  # Find first depth exceeding threshold
        ds['mld'] = -mask.idxmax("Z")  # idxmax finds 1st True if bool is used
    else:
        print("Illegal method specified")
        quit()

    return ds


if __name__ == "__main__":
    simulations = [
        'mrb_121_2', 'mrb_121_3', 'mrb_121_1']#, 'mrb_122_1', 'mrb_123_1', 'mrb_124_1',
        #'mrb_125_1', 'mrb_126_1', 'mrb_129', 'mrb_130', 'mrb_131',
        #'mrb_132', 'mrb_133', 'mrb_134', 'mrb_135', 'mrb_136',
        #'mrb_137', 'mrb_138', 'mrb_139', 'mrb_140_2', 'mrb_141',
        #'mrb_142', 'mrb_143', 'mrb_144', 'mrb_146']
    for runid in simulations:
        ze_tb(runid)
    
