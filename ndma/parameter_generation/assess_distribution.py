from turtledemo.penrose import start

import numpy as np
from networkx.algorithms import threshold

from ndma.parameter_generation.DSGRN_tools import *
import matplotlib.pyplot as plt
import frigidum
from frigidum.examples import tsp
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



def simulated_annealing_for_snake(points, plotting=False):
    dist = lambda x,y : np.linalg.norm(x-y)
    tsp.nodes = points
    tsp.nodes_count = points.shape[0]
    is_2d = np.shape(points)[1] == 2
    if is_2d:
        z = np.array([[complex(*c) for c in points]]) # notice the [[ ... ]]
        distance_matrix = abs(z.T-z)
    else:
        distance_matrix = np.empty([tsp.nodes_count, tsp.nodes_count])
        for i in range(tsp.nodes_count):
            for j in range(i + 1, tsp.nodes_count):
                distance_ij = dist(points[i,:],points[j,:])
                distance_matrix[i,j] = distance_ij
                distance_matrix[j,i] = distance_ij
    tsp.dist_eu = distance_matrix
    local_opt = frigidum.sa(random_start=tsp.random_start,
           objective_function=tsp.objective_function,
           neighbours=[tsp.euclidian_bomb_and_fix, tsp.euclidian_nuke_and_fix, tsp.route_bomb_and_fix, tsp.route_nuke_and_fix, tsp.random_disconnect_vertices_and_fix],
           copy_state=frigidum.annealing.naked,
           T_start=5,
           alpha=.8,
           T_stop=0.001,
           repeats=10**2,
           post_annealing = tsp.local_search_2opt)
    route = local_opt[0]
    if plotting:
        if is_2d:
            plt.scatter(points[:,0],points[:,1])

            for a, b in zip(route[:-1],route[1:]):
                x = points[[a,b]].T[0]
                y = points[[a,b]].T[1]
                plt.plot(x, y,c='r',zorder=-1)
            plt.gca().set_aspect('equal')
        if np.shape(points)[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(points[:,0],points[:,1],points[:,2])
            for a, b in zip(route[:-1],route[1:]):
                x = points[[a,b]].T[0]
                y = points[[a,b]].T[1]
                z = points[[a,b]].T[2]
                ax.plot(x, y, z)
            #ax.gca().set_aspect('equal')
    return route

def times_series(points, snake_indices):
    dist = lambda a,b: np.linalg.norm(points[a,:]-points[b,:])
    number_of_points = len(snake_indices)
    time_series = np.zeros(number_of_points-1)
    for i in range(number_of_points-1):
        time_series[i] = dist(snake_indices[i], snake_indices[i+1])
    return time_series


def snake_testing(points):

    def calculate_p_value(time_serie):
        median = np.median(time_serie)
        plt.plot(time_serie, color='blue')
        plt.plot(time_serie*0 + median, color='red')
        plt.show()
        short_than_mean = np.array([int(time_serie[i] < median) for i in range(len(time_serie))])
        shift_dif = short_than_mean[1:] - short_than_mean[:-1]
        all_1, = np.where(shift_dif == 1)
        all_min1, = np.where(shift_dif == -1)
        if all_1[0] < all_min1[0]:
            length_runs = all_min1 - all_1[:len(all_min1)]
        else:
            length_runs = all_1 - all_min1[:len(all_1)]
        length_runs = length_runs[length_runs>1]
        max_len = max(length_runs)
        average_len_runs = np.average(length_runs)
        n_runs = sum(length_runs>average_len_runs)

        p_value = 0
        return p_value


    snake_indices = simulated_annealing_for_snake(points, plotting=True)
    time_serie = times_series(snake_indices)
    p_value = calculate_p_value(time_serie)
    p_value = 1
    return p_value


def snake_sample(dim, repeat_size = 100, sample_size=100):
    known_distribution = np.zeros(repeat_size)
    for i in range(repeat_size):
        points = np.random.random((sample_size, dim))
        route = simulated_annealing_for_snake(points)
        time_serie = times_series(points, route)
        median = np.median(time_serie)
        plt.plot(time_serie, color='blue')
        plt.plot(time_serie * 0 + median, color='red')
        plt.show()
        longer_than_mean = np.array([int(time_serie[i] > median) for i in range(len(time_serie))])
        shift_dif = longer_than_mean[1:] - longer_than_mean[:-1]
        runs = []
        current_counter = 0
        for j in shift_dif:
            if j == 0:
                current_counter += 1
            elif current_counter > 0 :
                runs.append(current_counter)
                current_counter = 0
        max_len = max(runs)
        average_len_runs = np.average(np.array(runs))
        n_runs = sum(runs > average_len_runs)
        snake_ratio = n_runs/max_len
        if snake_ratio == 0:
            print('something weird, keep going, dim = ', dim)
        known_distribution[i] = snake_ratio
    return known_distribution



def simulated_annealing_test(size = 30, dim = 3):
    rand_points = np.random.random((30, 3))
    route = simulated_annealing_for_snake(rand_points, plotting=True)
    print(route)


if __name__ == '__main__':
    filename = 'snake_sample_results2.pickle'
    repeat_size = 2
    sample_size = 1000
    try:
        with open(filename, 'rb') as handle:
            dict_sample = pickle.load(handle)
    except FileNotFoundError:
        dict_sample = {}
    for dim in range(2,4):
        sample = snake_sample(dim, repeat_size=repeat_size, sample_size=sample_size)
        if (dim, sample_size) in dict_sample:
            dict_sample[(dim, sample_size)] = np.append(dict_sample[(dim, sample_size)],sample)
        else:
            dict_sample[(dim,sample_size)] = sample
        with open(filename, 'wb') as handle:
            pickle.dump(dict_sample, handle)
    print(dict_sample)