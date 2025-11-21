# Tool for making relax sponges/masks used with the RBCS package
# Modified from:
# https://xmitgcm.readthedocs.io/en/stable/demo_writing_binary_file.html

import numpy as np
import xmitgcm
import matplotlib.pylab as plt
import xmitgcm.file_utils
import xmitgcm.utils


def relax_mask(example_bin, shape, side_cells, side_max_M, bottom_cells,
               bottom_max_M, bin_name, scaling='exp'):
    """Creates a relax mask (M_rbc) based on an example input binary
    and, crucially, ts shape. The input binary can be anything 3D such
    as U, V, T, etc. The shape should be its dimension as a tuple, e.g.,
    (50, 100, 100), where the first dim is depth and the other dims are
    of course horizontal.

    Parameters:
        example_bin: File path to an example binary with desired shape
        shape: Tuple of three numbers describing the binary shape
        side_cells and bottom_cells: The number of non-zero cells along
            the boundary, i.e., the thickness of the sponge
        side_max_M and bottom_max_M: The values of the outermost cells
            (must be between 0 and 1).
        scaling: Describes how the sponge "strength" changes, 2 options:
            'exp' (default): The magnitude of the cells is halved
            every dx, e.g.,
                side_cells=2, side_max_M=1  =>  1,  0.5,     0,      0, 0,...
                side_cells=4, side_max_M=1  =>  1, 0.25, 0.125, 0.0625, 0,...
            'linear': The magnitude of the cells grows linearly, e.g.,
                side_cells=2, side_max_M=1  =>  1,  0.5,     0,      0, 0,...
                side_cells=4, side_max_M=1  =>  1, 0.75,   0.5,   0.25, 0,...

    Output:
        Saves a binary!                
    """

    # Simple check
    if side_max_M > 1 or bottom_max_M > 1 or side_max_M < 0 or bottom_max_M < 0:
        print('The side or bottom max value is out of bounds.')
        return

    # Creating lists of the cell values
    side_values = np.full(side_cells, side_max_M)
    bottom_values = np.full(bottom_cells, bottom_max_M)
    if scaling == 'exp':
        side_values = [i/2**n for n, i in enumerate(side_values)]
        bottom_values = [i/2**n for n, i in enumerate(bottom_values)]
    if scaling == 'linear':
        d_side, d_bottom = side_max_M/side_cells, bottom_max_M/bottom_cells
        side_values = [i-n*d_side for n,i in enumerate(side_values)]
        bottom_values = [i-n*d_bottom for n,i in enumerate(bottom_values)]

    # Opening an example field
    M = xmitgcm.utils.read_raw_data(
        example_bin, shape=shape, dtype=np.dtype('>f4'))
    M[:] = 0  # Interior M_rbc cells should be 0

    # Starting with the side cells; traverse in reverse to easily handle
    # overlaps in the corners
    if shape[1] < side_cells*2:  # i.e., 2D-adjacent domains
        for n, i in reversed(list(enumerate(side_values))):
            M[:, :, n], M[:, :, (-1)*n-1] = i, i
    elif shape[2] < side_cells*2:  # i.e., 2D-adjacent domains
        for n, i in reversed(list(enumerate(side_values))):
            M[:, n, :], M[:, (-1)*n-1, :] = i, i
    else:  # i.e., for fully 3D domains
        for n, i in reversed(list(enumerate(side_values))):
            M[:, n, :], M[:, (-1)*n-1, :] = i, i

    # Now do the bottom; in overlaps, defer to the higher value
    for n, i in reversed(list(enumerate(bottom_values))):
        M[(-1)*n-1, :, :] = np.where(
            M[(-1)*n-1, :, :] < i, i, M[(-1)*n-1, :, :])

    # Save the binary
    # For flattening: either...
    # np.moveaxis(M, [0, 1, 2], [-1, -2, -3]).flatten(order='F') or
    # M.flatten(order='C')
    xmitgcm.utils.write_to_binary(
        np.moveaxis(M, [0, 1, 2], [-1, -2, -3]).flatten(order='F'),
        '../MITgcm/so_plumes/binaries/' + bin_name)

    # # (Testing!) -- this needs to be set manually
    # M = xmitgcm.utils.read_raw_data(
    #     '../MITgcm/so_plumes/binaries/' + bin_name,
    #     shape=shape,
    #     dtype=np.dtype('>f4'))
    # X = np.linspace(0, 32, 33)
    # Y = np.linspace(0, 98, 99)
    # fig, ax = plt.subplots()
    # cs = ax.pcolormesh(X, Y, M[:, :, 0])
    # cbar = fig.colorbar(cs)
    # plt.savefig('relax_mask.png')


if __name__ == "__main__":
    example_bin = ('../../../home/robrow001/rjb_phd_paper1_code/' +
                   'binaries/bin_init_SA_33x594x99_variable.bin')
    shape = (99, 33, 594)
    side_cells = 50
    side_max_M = 1
    bottom_cells = 4
    bottom_max_M = 1
    bin_name = 'relax_mask_linear_33x594x99.bin'
    scaling = 'linear'
    relax_mask(example_bin, shape, side_cells, side_max_M, bottom_cells,
               bottom_max_M, bin_name, scaling)
