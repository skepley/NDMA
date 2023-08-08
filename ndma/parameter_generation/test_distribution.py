from ndma.parameter_generation.DSGRN_tools import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# test convergence of randomm walk to uniform distribution within a region


def index_k_neighbors(x, all_points, k=6):
    """
    INPUTS
    x              location of density computation - can be a point in all_points or not
    all_points     what we compute the density of
    k              using k nearest neighbors approach
    OUTPUT
    density_at_x
    """
    distance_of_points = np.array([np.linalg.norm(x - y) for y in all_points])
    index_of_neighbors = np.argpartition(distance_of_points, k)
    return index_of_neighbors[:k], distance_of_points


def local_density(x, all_points, k=10):
    """
    INPUTS
    x              location of density computation - can be a point in all_points or not
    all_points     what we compute the density of
    k              using k nearest neighbors approach
    OUTPUT
    density_at_x
    """
    index_of_neighbors, distance_of_points = index_k_neighbors(x, all_points, k)
    density_at_x = 1/ (np.average(distance_of_points[index_of_neighbors]))
    return density_at_x


def plot_heat(x, y, z):

    # define grid.
    xGrid = np.linspace(np.min(x), np.max(x), 100)
    yGrid = np.linspace(np.min(y), np.max(y), 100)
    # grid the data.
    zGrid = griddata((x, y), z, (xGrid[None, :], yGrid[:, None]), method='linear')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    zmin = np.nanmin(zGrid)
    zmax = np.nanmax(zGrid)
    palette = plt.matplotlib.colors.LinearSegmentedColormap('jet3', plt.cm.datad['jet'], 2048)
    palette.set_under(alpha=0.0)
    plt.imshow(zGrid, extent=(0, 3, 0, 3), cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='auto',
               interpolation='bilinear')

    plt.colorbar()  # draw colorbar
    return


def density_difference(many_x, points_0, points_1):
    """
    INPUTS
    many_x                    locations of density computation - array of points
    points_0, points_1        what we compare the density of
    OUTPUT
    difference_densities
    """
    loc_density_0 = np.array([local_density(x, points_0) for x in many_x])
    loc_density_1 = np.array([local_density(x, points_1) for x in many_x])
    #plot_heat(many_x[:,0],many_x[:,1],loc_density_0)
    plot_heat(many_x[:,0],many_x[:,1],loc_density_1)
    plt.show()
    plt.plot(range(len(loc_density_0)), np.sort(loc_density_1))
    plt.show()
    local_difference = np.array([np.abs(loc_density_0-loc_density_1) for x in many_x])
    difference_densities = np.max(local_difference)
    return difference_densities


def convergence_rate(points_0, points_1, selected_points=None):
    if selected_points is None:
        selected_points = points_1
    diff = density_difference(selected_points, points_0, points_1)
    return diff


def check_convergence(points_0, points_1, selected_points=None, threshold=10**-3):
    if selected_points is None:
        selected_points = points_1
    diff = convergence_rate(points_0, points_1, selected_points=selected_points)
    return diff < threshold




