# Script for giving an overview of the key results form the simulations

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import xmitgcm
import gsw


def results():
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

    # Filepaths
    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'
    S1 = fp + 'mrb_121'
    S2 = fp + 'mrb_122'
    S3 = fp + 'mrb_123'
    S4 = fp + 'mrb_124'
    S5 = fp + 'mrb_125'
    S6 = fp + 'mrb_119'

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters (since mpl uses inches)
    layout = [
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.', 'a5', 'a5', 'a5'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.', 'a5', 'a5', 'a5'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.', 'a5', 'a5', 'a5'],
        ['a1', 'a1', 'a2', 'a2', 'a3', 'a3', 'a4', 'a4', '.',  '.',  '.',  '.'],
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
    ds = open_dataset(S1)

    # Function for opening the diagnostics (easiest way to get the
    # lead area) --- useful for normalising the results but also
    # potentially misleading
    def get_lead_shape(fp, dt=4):
        ds = xmitgcm.open_mdsdataset(fp, prefix=['surfDiag'], delta_t=dt)
        lead_shape = ds['SFLUX'].isel(time=0)
        return lead_shape.where(lead_shape > 0)

    # Plot initial abs salinity
    ds['S'].isel(time=0, YC=10, XC=10).plot(y='Z', ax=ax1, c='k')

    # Shading
    z_ent = -112.97
    column = ds['S'].isel(time=0, YC=10, XC=10)
    da = column.sel(Z=slice(0, z_ent))
    y = np.append(da['Z'].to_numpy(), z_ent)
    x1 = np.append(da.to_numpy(), column.interp(Z=z_ent))
    x0 = x1[-1]
    ax1.fill_betweenx(y, x0, x1, color='grey', alpha=0.4)

    # Plot initial pot temp
    ds['T'].isel(time=0, YC=10, XC=10).plot(y='Z', ax=ax2, c='k')

    # Shading
    z_ent = -112.97
    column = ds['T'].isel(time=0, YC=10, XC=10)
    da = column.sel(Z=slice(0, z_ent))
    y = np.append(da['Z'].to_numpy(), z_ent)
    x1 = np.append(da.to_numpy(), column.interp(Z=z_ent))
    x0 = x1[0]
    ax2.fill_betweenx(y, x0, x1, color='grey', alpha=0.4)

    def plot_ax3_ax4_ax5(ds, lead_shape, label, lw=1.5, c='k'):

        # Plot heat content anomaly between start and end of forcing
        start = np.timedelta64(0, 'h')
        end = np.timedelta64(24, 'h')
        elapsed = end - start
        def calc_hc(ds):
            ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
            ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
            ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
            ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
            ds['HC'] = ds['rho']*ds['cp']*(ds['t_exact']-(-2))*ds['drF']*ds['rA']
            ds['HC_perlevel'] = ds['HC'].sum(['XC', 'YC'])
            return ds  # Units of HC are J (total per cell)
        ds = calc_hc(ds)
        da = (ds['HC_perlevel'].sel(time=end) -
              ds['HC_perlevel'].sel(time=start))
        (da/1e12/da['drF']).plot(y='Z', ax=ax3, c=c, linewidth=lw)

        # Plot the cumsum heat flux
        # Units: W /per m2 OF LEAD AREA/
        a = ds['rA'].where(lead_shape > 0).sum(['XC', 'YC']).values
        seconds = elapsed.astype('timedelta64[s]').astype(int)
        hf = np.cumsum(da[::-1])/(a*seconds)*(-1)  # W / m**2
        hf.plot(y='Z', ax=ax4, c=c, linewidth=lw)

        # Plot the entrainment HF (i.e., through a given depth) over time
        # This can be considered the net heat content change below the depth
        d = -51
        ds['hc_subset'] = ds['HC_perlevel'].sel(Z=slice(d, ds['Z'].min()))
        ds['HC_change'] = ds['hc_subset'].sum(dim='Z').diff(dim='time')
        ds['dt'] = ds['time'].diff('time').astype('timedelta64[s]').astype(int)
        hf_pertimestep = ds['HC_change']/(a*ds['dt'])
        hf_pertimestep['time'] = hf_pertimestep['time'].dt.total_seconds() / 3600
        smoothed = hf_pertimestep.rolling(time=5, center=True).mean()
        ax5.plot(hf_pertimestep['time'], smoothed*(-1), c=c, linewidth=lw,
                 label=label)

    plot_ax3_ax4_ax5(ds, get_lead_shape(S1), "S1")
    plot_ax3_ax4_ax5(open_dataset(S2), get_lead_shape(S2), "S2", lw=0.6, c='r')
    plot_ax3_ax4_ax5(open_dataset(S3), get_lead_shape(S3), "S3", lw=0.6, c='g')
    plot_ax3_ax4_ax5(open_dataset(S4), get_lead_shape(S4), "S4", lw=0.6, c='b')
    plot_ax3_ax4_ax5(open_dataset(S5), get_lead_shape(S5), "S5", lw=0.6, c='m')
    plot_ax3_ax4_ax5(open_dataset(S6), get_lead_shape(S6), "S6", lw=0.6, c='y')

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

    # Formatting the entrainment heat flux panel
    ax5.set_ylabel('')
    ax5.set_xlabel('')
    ax5.set_title('')
    ax5.grid()
    ax5.tick_params(axis='both', labelsize=8)

    # Adding legend
    ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2,
               prop={'size': 8}, frameon=False)

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
    ax5.set_title("(e) Entrainment\nheat flux", fontsize=8)
    ax5.set_xlabel("Model time (h)", fontsize=8)
    ax5.set_ylabel("W m$^{-2}$", fontsize=8, labelpad=0)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, wspace=0.5)
    plt.savefig("figure_results.pdf")
    plt.savefig("figure_results.svg", transparent=True)
    plt.savefig("figure_results.png", dpi=300)


if __name__ == "__main__":
    results()
