# Copied from: https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html

import numpy as np
import xmitgcm
import matplotlib.pylab as plt
import xmitgcm.file_utils
import xmitgcm.utils


def relax_mask(example_bin, Nx, Ny, Nr, side_cells, side_max_M, bottom_cells,
               bottom_max_M, bin_name, scaling='exp'):
    """ Creates a relax mask (M_rbc) using the specified geometry. The
    basic idea is that the sponges should be along the narrow side dims
    (i.e., not intersecting the lead) and also along the bottom. One or
    the other of the sides or the bottom sponges can be turned off by
    simply setting the associated max relax value to 0.
        Example dimensions are Nr=99, Nx=33, Ny=1188).
        side_cells and bottom_cells refer to the number of non-zero
    cells along the boundary (basically the thickness of the sponge).
        side_max_M and bottom_max_M refer to the values of the
    outermost cells (between 0 and 1).
        With scaling=exp, the magnitude of the cells is halved every
    dx (not really exponential but you get the point), e.g.,
     - side_cells=2, side_max_M=1  =>  1,  0.5,     0,      0, 0, 0
     - side_cells=4, side_max_M=1  =>  1, 0.25, 0.125, 0.0625, 0, 0 etc.
        With scaling=linear, the magnitude of the cells grows
    linearly, e.g.,
     - side_cells=2, side_max_M=1  =>  1,  0.5,     0,      0, 0, 0
     - side_cells=4, side_max_M=1  =>  1, 0.75,   0.5,   0.25, 0, 0 etc.
    """

    # Check to make sure the inputs will work
    if (side_max_M > 1 or bottom_max_M > 1 or side_max_M < 0 or
            bottom_max_M < 0):
        print('The side or bottom max value is out of bounds.')
        return

    # Creating lists of the cell values
    side_values = np.full(side_cells, side_max_M)
    # e.g., np.full creates arr w/ shape side_cells filled w/ value side_max_M
    bottom_values = np.full(bottom_cells, bottom_max_M)
    if scaling == 'exp':
        side_values = [i/2**n for n, i in enumerate(side_values)]
        bottom_values = [i/2**n for n, i in enumerate(bottom_values)]
    if scaling == 'linear':
        d_side, d_bottom = side_max_M/side_cells, bottom_max_M/bottom_cells
        side_values = [i-n*d_side for n, i in enumerate(side_values)]
        bottom_values = [i-n*d_bottom for n, i in enumerate(bottom_values)]

    # Create the basic array
    M = np.zeros((Nr, Nx, Ny), dtype='>f4')

    # Apply relaxation on the narrow horizontal dimension
    # Refactored from my original code by Chat because this is cleaner
    if Nx <= Ny:
        for n, i in reversed(list(enumerate(side_values))):
            M[:, :, n] = i
            M[:, :, -n-1] = i
    else:
        for n, i in reversed(list(enumerate(side_values))):
            M[:, n, :] = i
            M[:, -n-1, :] = i

    # Now do the bottom; if cases of overlapping, defer to the higher value
    for n, i in reversed(list(enumerate(bottom_values))):
        M[-n-1, :, :] = np.where(
            M[(-1)*n-1, :, :] < i, i, M[(-1)*n-1, :, :])

    # Save the binary
    xmitgcm.utils.write_to_binary(M.flatten(order='C'), bin_name)

    # (Testing!) -- needs to be set manually
    M = xmitgcm.utils.read_raw_data(
        bin_name, shape=(Nr, Nx, Ny), dtype=np.dtype('>f4'))
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(M[:, Nx//2, :])
    cbar = fig.colorbar(cs)
    plt.savefig('relax_mask.png')


def tmp_test_relax_mask():

    ds = xmitgcm.open_mdsdataset('/albedo/work/projects/p_so-clim/GCM_data/RowanMITgcm/mrb_141', prefix=['fluxDiag'], delta_t=4)
    ds['Z'] = ds['Z'].astype('<f4')

    ds['TOTUTEND'].isel(time=14, Z=15).plot.pcolormesh()
    plt.savefig('test.png')


if __name__ == "__main__":
    example_bin = 'bin_init_U_33x594x99.bin'
    Nr, Nx, Ny = 99, 33, 594
    side_cells = 25
    side_max_M = 1
    bottom_cells = 10
    bottom_max_M = 1
    bin_name = 'relax_mask_linear_sides_and_bottom_33x594x99.bin'
    scaling = 'linear'
    relax_mask(example_bin, Nx, Ny, Nr, side_cells, side_max_M, bottom_cells,
               bottom_max_M, bin_name, scaling)
