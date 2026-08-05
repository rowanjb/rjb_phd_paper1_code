# Script for giving an overview of the key results form the simulations

import f90nml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
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


# Calculte the vertical heat flux
def calc_hf(ds):

    # Calculate the heat content
    def calc_hc(ds, tref=-2):
        ds['P'] = gsw.p_from_z(ds['Z'], -69.0005)
        ds['t_exact'] = gsw.t_from_CT(ds['S'], ds['CT'], ds['P'])
        ds['cp'] = gsw.cp_t_exact(ds['S'], ds['t_exact'], ds['P'])
        ds['rho'] = gsw.rho(ds['S'], ds['CT'], ds['P'])
        ds['HC'] = ds['rho']*ds['cp']*(ds['CT']-tref)*ds['drF']*ds['rA']
        ds['HC_perlevel'] = ds['HC'].sum(['XC', 'YC'])
        return ds  # Units of HC are J (total per cell)

    # First calc the hc
    ds = calc_hc(ds)

    # Next define the start and end of forcing
    t1 = np.timedelta64(0, 'h')
    t2 = np.timedelta64(24, 'h')
    elapsed = t2 - t1

    # Calculare the HC change
    da = ds['HC_perlevel'].sel(time=t2) - ds['HC_perlevel'].sel(time=t1)

    # Next we want to calculate the cumsum heat flux
    # a = ds['rA'].where(get_lead_shape(S1) > 0).sum(['XC', 'YC']).values
    a = ds['rA'].sum(['XC', 'YC']).values  # Area
    seconds = elapsed.astype('timedelta64[s]').astype(int)
    hf = np.cumsum(da[::-1])/(seconds)*(-1)  # Units become W / m**2
    # Not we don't divide by area because we want to be able to compare
    # runs with (eg) different domains but same lead width/forcing.

    return hf/1e6  # Convert units to MJ


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
    fps = lambda run: fp + run

    # Simulation names
    sims_df = pd.read_csv(
        "paper_simulations.csv", sep=r"\s*,\s*", engine='python', dtype=str)
    sims_df = sims_df.set_index("Internal_name")

    # Initialise the plot
    cm = 1/2.54  # Inches to centimeters (since mpl uses inches)
    layout = [
        [ 'a1',  'a2',  'a3',  'a4'],
        [ 'a5',  'a6',  'a7',  'a8'],
        [ 'a9', 'a10', 'a11',   '.']]
    fig, axes = plt.subplot_mosaic(layout)
    ax1, ax2, ax3, ax4 = axes['a1'], axes['a2'], axes['a3'], axes['a4']
    ax5, ax6, ax7, ax8 = axes['a5'], axes['a6'], axes['a7'], axes['a8']
    ax9, ax10, ax11 = axes['a9'], axes['a10'], axes['a11']
    all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    fig.set_figwidth(18*cm)
    fig.set_figheight(22*cm)

    # Colours
    colours = [
        (0/256, 0/256, 0/256),
        (112/256, 160/256, 205/256),
        (196/256, 121/256, 0/256),
        (178/256, 178/256, 178/256),
    ]

    def plotting_function(runid, ax, colour, category, unit="", factor=1):
        char_value = str(sims_df[category][runid])
        if category == "w":
            char_value = str(
                int(sims_df[category][runid])*int(sims_df["dx"][runid]))
        elif category == "Sponges":
            if sims_df[category][runid] == '-':
                char_value = "None"
            elif sims_df[category][runid] == 'B':
                char_value = 'Bottom \nonly'
            elif sims_df[category][runid] == 'BS':
                char_value = 'Bottom +\nsides'
        if unit != "":
            unit = " " + unit
        label = (char_value + unit +
                 " (" + str(sims_df["Simulation"][runid]) + ")")
        if category == "dx":
            label = (char_value + unit + ", " + sims_df["z"][runid] + " levs"
                     " (" + str(sims_df["Simulation"][runid]) + ")")
        (calc_hf(open_dataset(fps(runid)))*factor).plot(
            y='Z', ax=ax, c=colour, label=label)

    # ax1
    ax = ax1
    category = "Smag_2D_factor"
    plotting_function('mrb_121_1', ax, colours[0], category)
    plotting_function('mrb_120_1', ax, colours[1], category)
    plotting_function('mrb_132', ax, colours[2], category)
    ax1.set_title("(a) 2D Smagorinsky\ncoefficient", fontsize=8)

    # ax2
    ax = ax2
    category = "Ar"
    plotting_function('mrb_139', ax, colours[0], category, "m$^2$ s$^{-1}$")
    plotting_function('mrb_120_1', ax, colours[1], category, "m$^2$ s$^{-1}$")
    plotting_function('mrb_130', ax, colours[2], category, "m$^2$ s$^{-1}$")
    plotting_function('mrb_122_1', ax, colours[3], category, "m$^2$ s$^{-1}$")
    ax2.set_title("(b) Vertical eddy\nviscosity", fontsize=8)

    # ax3
    ax = ax3
    category = "l_lead"
    plotting_function('mrb_123_1', ax, colours[0], category, "m", factor=0.5)
    plotting_function('mrb_130', ax, colours[1], category, "m", factor=1)
    plotting_function('mrb_125_1', ax, colours[2], category, "m", factor=2.08333)
    ax3.set_title("(c) Lead width\n(normalised), $l_{lead}$", fontsize=8)

    # ax4
    ax = ax4
    category = "q_surf"
    plotting_function('mrb_129', ax, colours[0], category, "W m$^{-2}$")
    plotting_function('mrb_130', ax, colours[1], category, "W m$^{-2}$")
    plotting_function('mrb_131', ax, colours[2], category, "W m$^{-2}$")
    ax4.set_title("(d) Heat flux,\n$q_{surf}$", fontsize=8)

    # ax5
    ax = ax5
    category = "Sponges"
    plotting_function('mrb_130', ax, colours[0], category)
    plotting_function('mrb_142', ax, colours[1], category)
    plotting_function('mrb_141', ax, colours[2], category)
    ax5.set_title("(e) Momentum\nsponges", fontsize=8)

    # ax6
    ax = ax6
    category = "dx"
    plotting_function('mrb_138', ax, colours[0], category, "m")
    plotting_function('mrb_137', ax, colours[1], category, "m")
    plotting_function('mrb_140_2', ax, colours[2], category, "m")
    plotting_function('mrb_146', ax, colours[3], category, "m")
    ax6.set_title("(f) Horizontal\nresolution, $\Delta x$", fontsize=8)

    # ax7
    ax = ax7
    category = "w"
    plotting_function('mrb_135', ax, colours[0], category, "m")
    plotting_function('mrb_130', ax, colours[1], category, "m")
    plotting_function('mrb_136', ax, colours[2], category, "m")
    ax7.set_title("(g) Domain\nwidth", fontsize=8)

    # ax8
    ax = ax8
    category = "T_freeze"
    plotting_function('mrb_121_1', ax, colours[0], category, "℃")
    plotting_function('mrb_124_1', ax, colours[1], category, "℃")
    plotting_function('mrb_134', ax, colours[2], category, "℃")
    ax8.set_title("(h) Freezing\npoint, $T_{freeze}$", fontsize=8)

    # ax9
    ax = ax9
    category = "a"
    plotting_function('mrb_130', ax, colours[0], category)
    plotting_function('mrb_133', ax, colours[1], category)
    plotting_function('mrb_126_1', ax, colours[2], category)
    ax9.set_title("(i) Stefan's Law\ncoefficient, $a$", fontsize=8)

    # ax10
    ax = ax10
    category = "T_freeze-T_air"
    plotting_function('mrb_144', ax, colours[0], category, "℃")
    plotting_function('mrb_130', ax, colours[1], category, "℃")
    plotting_function('mrb_143', ax, colours[2], category, "℃")
    ax10.set_title("(j) $T_{freeze}-T_{air}$\n#1", fontsize=8)

    # ax11
    ax = ax11
    category = "T_freeze-T_air"
    plotting_function('mrb_121_2', ax, colours[0], category, "℃")
    plotting_function('mrb_121_3', ax, colours[1], category, "℃")
    plotting_function('mrb_121_1', ax, colours[2], category, "℃")
    ax11.set_title("(k) "+r"$T_{freeze}-T_{air}$"+"\n#2", fontsize=8)

    for ax in all_axes:
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_ylim(-396, 0)
        ax.set_yticks([-396, -220, -125, -50, 0])
        ax.set_yticklabels([])
        ax.grid()
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlim(-5, 85)
        ax.set_xticks([0, 30, 60])
        ax.legend(
            fontsize=8,
            loc='lower center',
            frameon=True,
            facecolor='white',   # or another color
            edgecolor='none',
            framealpha=0.8,
            borderpad=0.05,      # padding between text and legend box
            labelspacing=0.2,   # vertical spacing between entries
            handletextpad=0.5,   # space between marker/line and text
            borderaxespad=0.2,  # space between legend and axes
            handlelength=1.0,  # line length
        )
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xticklabels([])
    for ax in [ax8, ax9, ax10, ax11]:
        ax.set_xlabel('TJ m$^{-1}$', fontsize=8)
    for ax in [ax1, ax5, ax9]:
        ax.set_yticklabels(['396 m', '220 m', '125 m', '50 m', '0 m'])

    plt.suptitle("Parameter dependence of mean vertical heat flux",
                 y=0.95, fontsize=8)

    plt.subplots_adjust(
        left=0.1, right=0.95, bottom=0.075, wspace=0.15, hspace=0.3)
    # plt.savefig("figures/figure_supp_results_1.pdf")
    plt.savefig("figures/figure_supp_results_2.svg", transparent=False)
    plt.savefig("figures/figure_supp_results_2.png", dpi=300)


if __name__ == "__main__":
    results()
