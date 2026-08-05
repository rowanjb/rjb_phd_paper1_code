# Script for giving an overview of the key results form the simulations

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import xmitgcm
import gsw


def results(runid, z_ent, tb):
    """Function for creating the 'results' figure."""

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
    colour = (112/256, 160/256, 205/256)

    # Filepaths
    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'
    sims = [
        'mrb_121', 'mrb_121_1', 'mrb_120_1', 'mrb_122_1', 'mrb_123_1',
        'mrb_124_1', 'mrb_125_1', 'mrb_126_1', 'mrb_129', 'mrb_130',
        'mrb_131', 'mrb_132', 'mrb_133', 'mrb_134', 'mrb_135',
        'mrb_136', 'mrb_137', 'mrb_138', 'mrb_139', 'mrb_140',
        'mrb_141', 'mrb_142', 'mrb_143', 'mrb_144']
    fps = [fp + sim for sim in sims]

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters (since mpl uses inches)
    layout = [
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.',  '.',  '.',  '.'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.', 'a5', 'a5', 'a5'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.', 'a5', 'a5', 'a5'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.', 'a5', 'a5', 'a5'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.',  '.',  '.',  '.']]
    fig, axes = plt.subplot_mosaic(layout)
    ax1, ax2, ax3 = axes['a1'], axes['a2'], axes['a3']
    ax4, ax5 = axes['a4'], axes['a5']
    fig.set_figwidth(18*cm)
    fig.set_figheight(7*cm)

    # Opening the model dataset
    def open_dataset(fp, dt=4):
        ds = xmitgcm.open_mdsdataset(fp, prefix=['S', 'T'], delta_t=dt)
        ds['Z'] = ds['Z'].astype('<f4')
        ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
        ds['sigma0'] = gsw.sigma0(ds['S'], ds['CT'])
        return ds
    ds = open_dataset(fps[0])

    # Plot initial abs salinity
    ds['S'].isel(time=0, YC=10, XC=10).plot(y='Z', ax=ax1, c='k')

    # Shading
    column = ds['S'].isel(time=0, YC=10, XC=10)  # Arbitrary location
    da = column.sel(Z=slice(0, z_ent))
    y = np.append(da['Z'].to_numpy(), z_ent)
    x1 = np.append(da.to_numpy(), column.interp(Z=z_ent))
    x0 = x1[-1]
    ax1.fill_betweenx(y, x0, x1, color='orange', alpha=0.4)

    # Plot initial pot temp
    ds['T'].isel(time=0, YC=10, XC=10).plot(y='Z', ax=ax2, c='k')

    # Shading
    column = ds['T'].isel(time=0, YC=10, XC=10)
    da = column.sel(Z=slice(0, z_ent))
    y = np.append(da['Z'].to_numpy(), z_ent)
    x1 = np.append(da.to_numpy(), column.interp(Z=z_ent))
    x0 = x1[0]  # It's x1[0] not tfreeze because this is the heat you unlock
    ax2.fill_betweenx(y, x0, x1, color='red', alpha=0.4)

    # Calculate the heat content
    def calc_hc(ds, tref=-1.9):
        # Note I tested different tref values (incl. -200), and there is
        # no visual change to the final figure. This is because the
        # anomaly calcs become
        #   hc2-hc1 = rho2*cp2*(t2-tref) - rho1*cp1*(t1-tref)
        #           = rho2*cp2*t2 - rho1*cp1*t1 + tref(-rho2*cp2+rho1*cp1)
        # where the last term is the only one that matters w/r/t tref, i.e.,
        # it is scaled by the density anomaly, which is tiny
        ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
        ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
        ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
        ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
        ds['HC'] = ds['rho']*ds['cp']*(ds['CT']-tref)*ds['drF']*ds['rA']
        ds['HC_perlevel'] = ds['HC'].sum(['XC', 'YC'])
        return ds  # Units of HC are J (total per cell)

    # Now we want the profiles that we'll plot in ax3, ax4, and ax5
    def calc_hc_and_hf(ds):

        # First calc the hc
        ds = calc_hc(ds)

        # Next define the start and end of forcing
        t1 = np.timedelta64(0, 'h')
        t2 = np.timedelta64(24, 'h')
        elapsed = t2 - t1

        # Then we can calculate the HC anomaly
        da = ds['HC_perlevel'].sel(time=t2) - ds['HC_perlevel'].sel(time=t1)
        hc_anom = da/1e12/da['drF']  # Units become TJ per m of depth

        # Next we want to calculate the cumsum heat flux
        # a = ds['rA'].where(get_lead_shape(S1) > 0).sum(['XC', 'YC']).values
        a = ds['rA'].sum(['XC', 'YC']).values  # Area
        seconds = elapsed.astype('timedelta64[s]').astype(int)
        hf = np.cumsum(da[::-1])/(a*seconds)*(-1)  # Units become W / m**2

        # Finally we also want the entrainment HF (i.e., through a
        # given depth) over time. This can be considered the net heat
        # content change below the depth. Note we use cumsum here so that
        # we can interpolate directly at the 50 m mooring sensor depth.
        reverse = slice(None, None, -1)
        ds['hc_cumsum'] = ds['HC_perlevel'].isel(
            Z=reverse).cumsum(dim="Z").isel(Z=reverse)
        ds['hc_under_50m'] = ds['hc_cumsum'].interp(Z=-50)
        ds['HC_change'] = ds['hc_under_50m'].diff(dim='time')
        ds['dt'] = ds['time'].diff('time').astype('timedelta64[s]').astype(int)
        hf_pertimestep = ds['HC_change']/(a*ds['dt'])
        hf_pertimestep['time'] = hf_pertimestep['time'].dt.total_seconds()/3600
        smoothed_ent_hf = hf_pertimestep.rolling(time=5, center=True).mean()
        smoothed_ent_hf = smoothed_ent_hf*(-1)

        return hc_anom, hf, smoothed_ent_hf

    # We now run calc_hc_and_hf for all sims, in order to get maxes and mins
    hc_anom, hf, smoothed_ent_hf = calc_hc_and_hf(ds)  # Start w/S1
    hc_anom = hc_anom.to_dataset(name=runid)
    hf = hf.to_dataset(name=runid)
    smoothed_ent_hf = smoothed_ent_hf.to_dataset(name=runid)
    for n, sim in enumerate(sims[1:]):
        if sim == 'mrb_138':
            new_ds = open_dataset(fps[n+1], dt=3)
        elif sim == 'mrb_140':
            new_ds = open_dataset(fps[n+1], dt=8)
        else:
            new_ds = open_dataset(fps[n+1])
        hc_anom[sim], hf[sim], smoothed_ent_hf[sim] = calc_hc_and_hf(new_ds)
    for profile in [hc_anom, hf, smoothed_ent_hf]:
        profile['max'] = profile.to_array().max("variable")
        profile['min'] = profile.to_array().min("variable")

    # Ax3
    # Note this is the heat anomaly for the full domain
    for sim in sims[1:]:
        hc_anom[sim].plot(ax=ax3, y='Z', c=colour, lw=0.5)
    ax3.fill_betweenx(
        hc_anom['Z'].to_numpy(),
        hc_anom['min'].to_numpy(),
        hc_anom['max'].to_numpy(),
        color=colour, alpha=0.4, ec='none')
    hc_anom[runid].plot(ax=ax3, y='Z', c='k')

    # Ax4
    # Note this is the heat flux per m2 over the full domain, ie smaller
    # domains have higher heat fluxes;
    for sim in sims[1:]:
        hf[sim].plot(ax=ax4, y='Z', c=colour, lw=0.5)
    ax4.fill_betweenx(
        hf['Z'].to_numpy(),
        hf['min'].to_numpy(),
        hf['max'].to_numpy(),
        color=colour, alpha=0.4, ec='none')
    hf[runid].plot(ax=ax4, y='Z', c='k')

    # Ax5
    for sim in sims[1:]:
        smoothed_ent_hf[sim].plot(ax=ax5, c=colour, lw=0.5)
    ax5.fill_between(
        smoothed_ent_hf['time'].to_numpy(),
        smoothed_ent_hf['min'].to_numpy(),
        smoothed_ent_hf['max'].to_numpy(),
        color=colour, alpha=0.4, ec='none')
    smoothed_ent_hf[runid].plot(ax=ax5, c='k')

    # Formatting the "profile" panels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(-396, 0)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_yticks([-396, -220, -125, -50, 0])
        ax.set_yticklabels([])
        ax.grid(axis='y')
        ax.set_title('')
        ax.tick_params(axis='both', labelsize=8)
    ax1.set_yticklabels(['396 m', '220 m', '125 m', '50 m', '0 m'])
    ax3.set_xticks([-0.3, 0, 0.3])
    ax4.set_xticks([0, 250, 500])

    # Formatting the entrainment heat flux panel
    ax5.set_ylabel('')
    ax5.set_xlabel('')
    ax5.set_title('')
    ax5.grid()
    ax5.tick_params(axis='both', labelsize=8)
    ax5.set_xticks([12, 24, 36, 48, 60])

    # Adding words etc
    ax1.set_title("(a) Absolute\nsalinity", fontsize=8)
    ax1.set_xlabel("g kg$^{-1}$", fontsize=8)
    ax2.set_title("(b) Potential\ntemperature, "+r"$\theta$", fontsize=8)
    ax2.set_xlabel("℃", fontsize=8)
    ax3.set_title(
        "(c) Heat anomaly,\n"+r"$HC_{24\ \mathrm{h}}-HC_{0\ \mathrm{h}}$",
        fontsize=8)
    ax3.set_xlabel("TJ m$^{-1}$", fontsize=8)
    ax4.set_title("(d) Mean vert.\nheat flux", fontsize=8)
    ax4.set_xlabel("W m$^{-2}$", fontsize=8)
    ax5.set_title("(e) Entrainment\nheat flux at 50 m", fontsize=8)
    ax5.set_xlabel("Model time (h)", fontsize=8)
    ax5.set_ylabel("W m$^{-2}$", fontsize=8, labelpad=0)

    # Add maximum heat flux
    maxes = hf.max()
    maxes_np = [maxes[i].to_numpy() for i in list(maxes.keys())]
    p = abs(min([maxes[runid].to_numpy() - i for i in maxes_np]))
    m = abs(max([maxes[runid].to_numpy() - i for i in maxes_np]))
    ax4.text(
        0.01, 0.3,
        (r"$HF_{max}=$"+"\n"+str(np.round(maxes[runid].to_numpy(), 1)) +
         r"$^{+"+str(int(np.round(p, 0)))+r"}_{-"+str(int(np.round(m, 0))) +
         r"}$"+"\n"+"W m$^{-2}$"),
        transform=ax4.transAxes, fontsize=8, color='k')

    # Add entrainment depth annotation
    ax1.text(
        0.01, 0.5, "Entrainment\ndepth =\n" + str(round((-1)*z_ent, 1)) + " m",
        transform=ax1.transAxes, fontsize=8, color='darkorange')

    # Add entrainment heat annotation
    ax2.text(
        0.01, 0.5, "Thermal\nbarrier =\n" + str(tb) + "\nMJ m$^{-2}$",
        transform=ax2.transAxes, fontsize=8, color='red')

    plt.subplots_adjust(
        left=0.1, right=0.95, bottom=0.15, wspace=0.5, hspace=0.5)
    # plt.savefig("figures/figure_results.pdf")
    plt.savefig("figures/figure_results.svg", transparent=False)
    plt.savefig("figures/figure_results.png", dpi=300)


if __name__ == "__main__":
    results('mrb_121', -108.5, 112.8)
