# Probably going to end up being deleta-able

import xmitgcm
import matplotlib.pyplot as plt
import numpy as np
import gsw


proj_dirs = [
    "../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/",
    "../MITgcm/so_plumes/"]
runs = [
    'mrb_084', 'mrb_085', 'mrb_086', 'mrb_087', 'mrb_088', 'mrb_089',
    'mrb_090', 'mrb_091', 'mrb_092']
delta_ts = {
    'mrb_084': 2, 'mrb_085': 4, 'mrb_086': 4, 'mrb_087': 4, 'mrb_088': 4,
    'mrb_089': 4, 'mrb_090': 4, 'mrb_091': 4, 'mrb_092': 4}
delta_xs = {
    'mrb_084': 2, 'mrb_085': 4, 'mrb_086': 4, 'mrb_087': 4, 'mrb_088': 4,
    'mrb_089': 4, 'mrb_090': 4, 'mrb_091': 4, 'mrb_092': 4}


def plot_ice_interface_hovm(run):

    dt, dx = delta_ts[run], delta_xs[run]
    try:
        fp = proj_dirs[0]+run
        ds = xmitgcm.open_mdsdataset(
            fp, prefix=['S', 'T'], delta_t=dt, ref_date="2021-9-13 12:0:0")
    except:
        fp = proj_dirs[1]+run
        ds = xmitgcm.open_mdsdataset(
            fp, prefix=['S', 'T'], delta_t=dt, ref_date="2021-9-13 12:0:0")

    # Take surface mean temp in the along-lead direction
    da = ds['T'].isel(Z=0).mean('YC')

    # Plot
    fig, ax = plt.subplots()
    da.plot.pcolormesh(y='time', ax=ax, vmin=-1.87, vmax=-1.57)
    ax.set_xlabel("Across-sea ice lead direction")
    ax.set_ylabel("")
    ax.set_xlim(
        int(np.floor(ds['XC'].max().values/2))-197,
        int(np.floor(ds['XC'].max().values/2))+197,
        )
    plt.xticks(rotation=30)
    plt.tight_layout()
    ax.set_title(
        "Mean surface pot. temp. Hovmöller ("+run+") dx="+str(dx)+" m")
    plt.savefig("surface_temp_"+run+".png", dpi=600)


def plot_pt_depth():

    def calc_hc_depth_levels(ds):
        ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
        ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
        ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
        ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
        ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
        ds['HC'] = ds['rho']*ds['cp']*(ds['t_exact']-(-2))*ds['drF']*ds['rA']
        ds['HC'] = ds['HC'].sum(['XC', 'YC'])
        ds["HC'"] = ds['HC'] - ds['HC'].isel(time=0)
        return ds

    fig, ax = plt.subplots()
    for run in runs:
        dt, dx = delta_ts[run], delta_xs[run]
        try:
            fp = proj_dirs[0]+run
            ds = xmitgcm.open_mdsdataset(
                fp, prefix=['S', 'T'], delta_t=dt, ref_date="2021-9-13 12:0:0")
        except:
            fp = proj_dirs[1]+run
            ds = xmitgcm.open_mdsdataset(
                fp, prefix=['S', 'T'], delta_t=dt, ref_date="2021-9-13 12:0:0")
        ds = calc_hc_depth_levels(ds)
        label = run+" "+str(ds["HC'"].sum().values/1e12)[:5]+" TJ"
        ds["HC'"].isel(time=-1).plot(y='Z', label=label)
    ax.set_ylabel("Depth ($m$)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.legend()
    ax.set_title(r"$HC_{end}-HC_{start}$")
    plt.savefig("water_col_HC_anom.png", dpi=600)


def plot_surface():
    """Temporary"""

    ds = xmitgcm.open_mdsdataset(
        "../MITgcm/so_plumes/mrb_092",
        prefix=['S', 'T'],
        delta_t=4,
        ref_date="2021-9-13 12:0:0")

    fig, ax = plt.subplots()
    fig.set_figheight(1)
    fig.set_figwidth(8)
    c = ds['T'].isel(time=2, Z=0).plot.contourf(
        Y='Z', levels=20, cmap='Blues_r', vmin=-2, vmax=0,
        add_colorbar=False)
    ax.set_title("")
    cbar = plt.colorbar(c, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_xlabel('Abs. salinity ($g$ $kg^{-1}$)',
                       fontdict={'fontsize': '9'})
    ax.tick_params(axis='both', labelsize=9)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([0, 66, 132])  # -50, -125, -220, -396])
    ax.set_yticklabels(["0 m", "66 m", "132 m"])  # ["50 m", "125 m", "220 m", "396 m"])
    ax.set_xticks([])  # 0, 1138, 1238, 2376])
    # ax.set_xticklabels(["0 km", "1.14 km", "1.24 km", "2.38 km"])
    plt.savefig('old_figs_and_scripts/horiz_salt_srfc.svg',
                transparent=True)


if __name__ == "__main__":
    # for run in runs:
    #     plot_ice_interface_hovm(run)
    # plot_pt_depth()
    plot_surface()
