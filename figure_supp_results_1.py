# Script for giving an overview of the key results form the simulations

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import xmitgcm
import gsw


# Opening the model dataset
def open_dataset(fp, dt=4):
    ds = xmitgcm.open_mdsdataset(fp, prefix=['S', 'T'], delta_t=dt)
    ds['Z'] = ds['Z'].astype('<f4')
    ds['CT'] = gsw.CT_from_pt(ds['S'], ds['T'])
    ds['sigma0'] = gsw.sigma0(ds['S'], ds['CT'])
    return ds


# Calculte the vertical heat flux
def calc_hc(ds):

    # Calculate the heat content
    def calc_h(ds, tref=-2):
        ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
        ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
        ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
        ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
        ds['HC'] = ds['rho']*ds['cp']*(ds['CT']-tref)*ds['drF']*ds['rA']
        ds['HC_perlevel'] = ds['HC'].sum(['XC', 'YC'])
        return ds  # Units of HC are J (total per cell)

    # First calc the hc
    ds = calc_h(ds)

    # Next define the start and end of forcing
    t1 = np.timedelta64(0, 'h')
    t2 = np.timedelta64(24, 'h')

    # Calculare the HC change
    da = ds['HC_perlevel'].sel(time=t2) - ds['HC_perlevel'].sel(time=t1)
    hc_anom = da/1e12/da['drF']  # Units become TJ per m of depth

    return hc_anom


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
    colour = (112/256, 160/256, 205/256)

    # Filepaths
    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/'
    fps = lambda run: fp + 'mrb_' + run
    #sims = list(fps.keys())

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters (since mpl uses inches)
    layout = [
        ['a1', 'a2', 'a3', 'a4', 'a5'],
        ['a6', 'a7', 'a8', 'a9', 'a10']]
    fig, axes = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = axes['a1'], axes['a2'], axes['a3'], axes['a4']
    ax5, ax6, ax7, ax8 = axes['a5'], axes['a6'], axes['a7'], axes['a8']
    ax9, ax10 = axes['a9'], axes['a10']
    all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    fig.set_figwidth(18*cm)
    fig.set_figheight(15*cm)

    # Colours
    colours = [
        (0/256, 0/256, 0/256),
        (112/256, 160/256, 205/256),
        (196/256, 121/256, 0/256),
        (178/256, 178/256, 178/256),
    ]

    # ax1
    ax = ax1
    calc_hc(open_dataset(fps('132'))).plot(y='Z', ax=ax, c=colours[0], label='2.2')
    calc_hc(open_dataset(fps('120'))).plot(y='Z', ax=ax, c=colours[1], label='3.1')
    calc_hc(open_dataset(fps('121'))).plot(y='Z', ax=ax, c=colours[2], label='4.0')

    # ax2
    ax = ax2
    calc_hc(open_dataset(fps('139'))).plot(y='Z', ax=ax, c=colours[0], label='2E-3 m$^2$ s$^{-1}$')
    calc_hc(open_dataset(fps('120'))).plot(y='Z', ax=ax, c=colours[1], label='2E-4 m$^2$ s$^{-1}$')
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[2], label='2E-5 m$^2$ s$^{-1}$')
    calc_hc(open_dataset(fps('122'))).plot(y='Z', ax=ax, c=colours[3], label='2E-6 m$^2$ s$^{-1}$')

    # ax3
    ax = ax3
    (calc_hc(open_dataset(fps('123')))/2).plot(y='Z', ax=ax, c=colours[0], label='200 m')
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[1], label='100 m')
    (calc_hc(open_dataset(fps('125')))/0.48).plot(y='Z', ax=ax, c=colours[2], label='48 m')

    # ax4
    ax = ax4
    calc_hc(open_dataset(fps('129'))).plot(y='Z', ax=ax, c=colours[0], label='0 W')
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[1], label='200 W')
    calc_hc(open_dataset(fps('131'))).plot(y='Z', ax=ax, c=colours[2], label='1000 W')

    # ax5
    ax = ax5
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[0], label='None')
    calc_hc(open_dataset(fps('141'))).plot(y='Z', ax=ax, c=colours[1], label='Bottom +\nsides')
    calc_hc(open_dataset(fps('142'))).plot(y='Z', ax=ax, c=colours[2], label='Bottom\nonly')

    # ax6
    ax = ax6
    calc_hc(open_dataset(fps('140_2'))).plot(y='Z', ax=ax, c=colours[0], label='$dx=$8 m\n99 levels')
    calc_hc(open_dataset(fps('145'), dt=8)).plot(y='Z', ax=ax, c='b', label='$dx=$16 m\n45 levels')
    calc_hc(open_dataset(fps('146'), dt=8)).plot(y='Z', ax=ax, c='b', label='$dx=$16 m\n99 levels')
    calc_hc(open_dataset(fps('137'))).plot(y='Z', ax=ax, c=colours[1], label='$dx=$4 m\n99 levels')
    calc_hc(open_dataset(fps('138'), dt=3)).plot(y='Z', ax=ax, c=colours[2], label='$dx=$2 m\n198 levels')

    # ax7
    ax = ax7
    calc_hc(open_dataset(fps('135'))).plot(y='Z', ax=ax, c=colours[0], label='1188 m')
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[1], label='2376 m')
    calc_hc(open_dataset(fps('136'))).plot(y='Z', ax=ax, c=colours[2], label='4752 m')

    # ax8
    ax = ax8
    calc_hc(open_dataset(fps('121'))).plot(y='Z', ax=ax, c=colours[0], label='-1.9 ℃')
    calc_hc(open_dataset(fps('124'))).plot(y='Z', ax=ax, c=colours[1], label='-1.87 ℃')
    calc_hc(open_dataset(fps('134'))).plot(y='Z', ax=ax, c=colours[2], label='-1.84 ℃')

    # ax9
    ax = ax9
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[0], label='0.037')
    calc_hc(open_dataset(fps('133'))).plot(y='Z', ax=ax, c=colours[1], label='0.034')
    calc_hc(open_dataset(fps('126'))).plot(y='Z', ax=ax, c=colours[2], label='0.031')

    # ax10
    ax = ax10
    calc_hc(open_dataset(fps('144'))).plot(y='Z', ax=ax, c=colours[0], label=r'13.25 ℃')
    calc_hc(open_dataset(fps('130'))).plot(y='Z', ax=ax, c=colours[1], label=r'17.25 ℃')
    calc_hc(open_dataset(fps('143'))).plot(y='Z', ax=ax, c=colours[2], label=r'21.25 ℃')

    for ax in all_axes:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_ylim(-396, 0)
        ax.set_yticks([-396, -220, -125, -50, 0])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_title('')
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlim(-0.28, 0.28)
        ax.set_xticks([-0.2, 0, 0.2])
        ax.legend(
            fontsize=8,
            loc='lower center',
            frameon=True,
            facecolor='white',   # or another color
            edgecolor='none',
            framealpha=0.68,
            borderpad=0.1,      # padding between text and legend box
            labelspacing=0.2,   # vertical spacing between entries
            handletextpad=0.5,   # space between marker/line and text
            borderaxespad=0.2,  # space between legend and axes
        )
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticklabels([])
    for ax in [ax6, ax7, ax8, ax9, ax10]:
        ax.set_xlabel('TJ m$^{-1}$', fontsize=8)
    ax1.set_yticklabels(['396 m', '220 m', '125 m', '50 m', '0 m'])
    ax6.set_yticklabels(['396 m', '220 m', '125 m', '50 m', '0 m'])

    plt.suptitle("Parameter dependence of mean vertical heat flux", fontsize=8)
    ax1.set_title("(a) 2D Smagorinsky\ncoefficient", fontsize=8)
    ax2.set_title("(b) Vertical eddy\nviscosity", fontsize=8)
    ax3.set_title("(c) Lead\nwidth*", fontsize=8)
    ax4.set_title("(d) Surface heat\nflux, $Q$", fontsize=8)
    ax5.set_title("(e) Relaxation\nsponges", fontsize=8)
    ax6.set_title("(f) Grid\nresolution", fontsize=8)
    ax7.set_title("(g) Domain\nwidth", fontsize=8)
    ax8.set_title("(h) Freezing\npoint", fontsize=8)
    ax9.set_title("(i) Stefan's Law\ncoefficient, $a$", fontsize=8)
    ax10.set_title("(j) Stefan's Law\n"+r"Ocean-atmos. $\Delta$T", fontsize=8)

    plt.subplots_adjust(
        left=0.1, right=0.95, bottom=0.075, wspace=0.15, hspace=0.25)
    # plt.savefig("figures/figure_supp_results_1.pdf")
    plt.savefig("figures/figure_supp_results_1.svg", transparent=False)
    plt.savefig("figures/figure_supp_results_1.png", dpi=300)


def calcs_for_SCAR():
    """Deletable"""

    # Filepaths
    fp = '../../../work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_121'

    da = calc_hc(open_dataset(fp))
    da = da*da['drF']  # Convert to heat per layer
    da = da.cumsum()
    print(da.interp(Z=-113).to_numpy())
    #print(da.where(da.Z > -113, drop=True).sum().to_numpy())


if __name__ == "__main__":
    #results()
    calcs_for_SCAR()
