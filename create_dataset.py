"""
Some code to create and manage a huge data set of stored parameters, created a priori, then accessed as needed


Author: Elena Queirolo
Created: 1st March 2021
Modified: 23rd May 2024
"""
from hill_model import *
import numpy as np
import random
# from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from datetime import datetime
import warnings
from models.TS_model import ToggleSwitch
import matplotlib.pyplot as plt
import DSGRN
from DSGRN_functionalities import par_to_region_wrapper, from_string_to_Hill_data, par_to_region, \
    from_region_to_deterministic_point
from toggle_switch_heat_functionalities import fiber_sampler, parameter_to_DSGRN_coord
import datetime


def oneregion_dataset(f_hill_model, parameter_region, dataset_size: int, network, n_parameters, filename=None,
                      save_file=True, optimize=True):
    """
    INPUT
    f_hill_model        Hill model class
    parameter_regions   vector of two integers, indicating the two parameter regions of interest in DSGRN
    dataset_size        integer, defining the size of the dataset to be created
    filename            string, the name of the dataset file
    network             DSGRN Network describing the network structure
    n_parameters        integer, the number of NDMA parameters (ex: 42 for the restricted EMT model)

    OUTPUT
    best_score          float, indicating how good the found distribution is: 1 - 50% of points in each region,
                        0.5 - smallest region only has 25% of points, 0 - smallest region is never sampled
    best_coef           vector, coefficients defining the used Gaussian distribution

    RESULT
    filename file is created such that:

    file_name           name of the saved file storing three items:
        optimal_coef        the coefficients used to created the dataset
        data                the data itself
        parameter_region    an integer vector having values 0,1 or 2 and length equal to data, indicating in which
                            DSGRN region each element of data lives in - first region, second region, neither.

    The algorithm creates the default points in the two regions, and uses them as base to create a gaussian cloud around
    them. If the two initial points are not ideal, other closer to the boundary between the two regions are chosen, in
    the hope of improving the starting data cloud
    The Gaussian cloud is then 'optimised' by randomly tweaking its coefficients to better distribute the points it
    generates
    """
    # rank datasets acording to score
    bin_size = lambda vec: np.array([np.sum(vec == j) for j in range(1)])

    def score(coefs):
        data_vec = ND_sampler(coefs[:n_parameters], coefs[n_parameters:], 500)
        parameter_region_vec = assign_region(data_vec)
        bins = bin_size(parameter_region_vec)
        scor = min(bins) * len(bins) / np.size(parameter_region_vec)
        return scor

    def from_point_to_coefs(a):
        mu = a
        Sigma = 0.001*np.eye(np.size(a), np.size(a))
        coef_ab = np.append(mu, Sigma.flatten())
        return coef_ab

    ND_sampler = distribution_sampler()
    parameter_graph = DSGRN.ParameterGraph(network)

    # sampling from each region
    pars0, sources_vec, targets_vec = from_region_to_deterministic_point(network, parameter_region)

    assign_region = par_to_region_wrapper(f_hill_model, parameter_region, parameter_graph, sources_vec, targets_vec)

    # Create initial distribution
    initial_coef = from_point_to_coefs(pars0)
    initial_score = score(initial_coef)

    if initial_score < 0.1:
        warnings.warn(
            'The initial Gaussian distribution chosen is very poor, likely low quality results to be expected')
        print('Initial score = ', initial_score)

    if optimize:
        best_score, best_coef = optimize_wrt_score(initial_coef, score)
        if best_score < 0.2:
            warnings.warn('Poor quality of the final distribution, consider choosing other starting points')
    else:
        best_score, best_coef = initial_score, initial_coef

    if save_file:
        if filename is None:
            filename = "region_" + str(parameter_region) + datetime.datetime.now().strftime("_date%d_%m_%Y")
        _ = generate_datafile_from_coefs(filename, best_coef, ND_sampler, assign_region, dataset_size, n_parameters)

    return best_score, best_coef


def tworegions_dataset(f_hill_model, parameter_regions, dataset_size: int, n_parameters, network, filename=None,
                       save_file=True):
    """
    INPUT
    f_hill_model        Hill model class
    parameter_regions   vector of two integers, indicating the two parameter regions of interest in DSGRN
    dataset_size        integer, defining the size of the dataset to be created
    filename            string, the name of the dataset file
    network             DSGRN Network describing the network structure
    n_parameters        integer, the number of NDMA parameters (ex: 42 for the restricted EMT model)

    OUTPUT
    best_score          float, indicating how good the found distribution is: 1 - 50% of points in each region,
                        0.5 - smallest region only has 25% of points, 0 - smallest region is never sampled
    best_coef           vector, coefficients defining the used Gaussian distribution

    RESULT
    filename file is created such that:

    file_name           name of the saved file storing three items:
        optimal_coef        the coefficients used to created the dataset
        data                the data itself
        parameter_region    an integer vector having values 0,1 or 2 and length equal to data, indicating in which
                            DSGRN region each element of data lives in - first region, second region, neither.

    The algorithm creates the default points in the two regions, and uses them as base to create a gaussian cloud around
    them. If the two initial points are not ideal, other closer to the boundary between the two regions are chosen, in
    the hope of improving the starting data cloud
    The Gaussian cloud is then 'optimised' by randomly tweaking its coefficients to better distribute the points it
    generates
    """
    # rank datasets acording to score
    bin_size = lambda vec: np.array([np.sum(vec == j) for j in range(2)])

    def score(coefs):
        data_vec = ND_sampler(coefs[:n_parameters], coefs[n_parameters:], 500)
        parameter_region_vec = assign_region(data_vec)
        bins = bin_size(parameter_region_vec)
        scor = min(bins) * len(bins) / np.size(parameter_region_vec)
        return scor

    def from_points_to_coefs(a, b):
        Sigma, mu = normal_distribution_around_points(np.array([a]), np.array([b]))
        coef_ab = np.append(mu, Sigma.flatten())
        return coef_ab

    ND_sampler = distribution_sampler()
    parameter_graph = DSGRN.ParameterGraph(network)

    # sampling from each region
    pars0, sources_vec, targets_vec = from_region_to_deterministic_point(network, parameter_regions[0])
    pars1, _, _ = from_region_to_deterministic_point(network, parameter_regions[1])

    assign_region = par_to_region_wrapper(f_hill_model, parameter_regions, parameter_graph, sources_vec, targets_vec)

    # Create initial distribution
    initial_coef = from_points_to_coefs(pars0, pars1)
    initial_score = score(initial_coef)

    # trying to get more points in missing region
    # looking for "middle point" between region 0 and 1
    if initial_score == 0:
        middle_point = (pars1 + pars0) / 2
        existing_region = par_to_region(f, middle_point, parameter_regions, parameter_graph, sources_vec,
                                        targets_vec)
        for i in range(10):
            middle_point = (pars1 + pars0) / 2
            if par_to_region(f, middle_point, parameter_regions, parameter_graph, sources_vec,
                             targets_vec) == existing_region:
                if existing_region == 0:
                    pars0 = middle_point
                else:
                    pars1 = middle_point
            else:
                print(i, 'iterations of bisection done to move the monostable pars closer to the bistable one')
                break
        initial_coef = from_points_to_coefs(pars0, pars1)
        initial_score = score(initial_coef)

    if initial_score < 0.1:
        warnings.warn(
            'The initial Gaussian distribution chosen is very poor, likely low quality results to be expected')

    best_score, best_coef = optimize_wrt_score(initial_coef, score)

    if best_score < 0.2:
        warnings.warn('Poor quality of the final distribution, consider choosing other starting points')

    if save_file:
        if filename is None:
            filename = "tworegions_" + str(parameter_regions[0]) + "_" + str(parameter_regions[1]) + \
                       datetime.datetime.now().strftime("_date%d_%m_%Y")
        _ = generate_datafile_from_coefs(filename, best_coef, ND_sampler, assign_region, dataset_size, n_parameters)

    return best_score, best_coef


def optimize_wrt_score(initial_val, score, iters=100):
    best_value = initial_val
    best_score = score(best_value)

    for iteri in range(iters):
        random_val = best_value * (1 + np.random.rand(np.size(initial_val)) * 0.05)
        random_score = score(random_val)
        if random_score > best_score:
            best_value = random_val
            best_score = random_score
    return best_score, best_value


def create_dataset(n_parameters: int, assign_region, n_parameter_region: int, size_dataset: int, file_name=None,
                   initial_coef=None):
    """
    create_dataset uses the information concerning a Hill model and its number of parameter regions to create a Fisher
    distribution spanning the parameter space such that all parameter regions are similarly sampled.
    Once the Fisher distribution is found, a sample of the distribution is taken. All information is then stored in a
    npz file.

    At the moment it ony works for the Toggle Switch

    INPUT
    n_parametes         interger, number of parameters of the semi-algebraic set
    assign_region       function, takes as input a parameter of an array of parameters and returns (an array of) region(s)
    n_parameter_region  integer, how many parameter regions are associated to the model
    size_dataset        integer, size of the output dataset
    file_name           string, name of the saved file

    OUTPUT
    file_name           name of the saved file

    helper functions:
    distribution_sampler
    DSGRN_parameter_region
    generate_datafile_from_coefs
    """
    warnings.warn('This function is deprecated, please use the new version instead')
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        file_name = f"{timestamp}" + '.npz'

    sampler_global = distribution_sampler()
    sampler_fisher = region_sampler_fisher()

    def sampler_score_fisher(fisher_coefficients):

        data_sample = sampler_fisher(fisher_coefficients[:n_parameters], fisher_coefficients[n_parameters:],
                                     5 * 10 ** 3)
        data_region = assign_region(data_sample)
        # TODO: link to DSGRN, this takes as input a matrix of parameters par[1:n_pars,1:size_sample], and returns a
        # vector data_region[1:size_sample], such that data_region[i] tells us which region par[:, i] belongs to
        # data_region goes from 0 to n_parameter_region -1
        counter = np.zeros(n_parameter_region)
        for iter_loc in range(n_parameter_region):
            counter[iter_loc] = np.count_nonzero(data_region == iter_loc)
        score = 1 - np.min(counter) / np.max(counter)
        # print(score) # lowest score is best score!
        return score  # score must be minimized

    def sampler_score(normal_coefficients):

        data_sample = sampler_global(normal_coefficients[:n_parameters], normal_coefficients[n_parameters:],
                                     5 * 10 ** 3)
        data_region = assign_region(data_sample)
        # TODO: link to DSGRN, this takes as input a matrix of parameters par[1:n_pars,1:size_sample], and returns a
        # vector data_region[1:size_sample], such that data_region[i] tells us which region par[:, i] belongs to
        # data_region goes from 0 to n_parameter_region -1
        counter = np.zeros(n_parameter_region)
        for iter_loc in range(n_parameter_region):
            counter[iter_loc] = np.count_nonzero(data_region == iter_loc)
        score = 1 - np.min(counter) / np.max(counter)
        # print(score) # lowest score is best score!
        return score  # score must be minimized

    size_coef = n_parameters * (1 + n_parameters)
    # for fisher  size_coef = 2*n_parameters
    if initial_coef is None:
        coefficients = np.abs(np.random.normal(size=size_coef))
    elif len(initial_coef) != size_coef:
        coefficients = np.abs(np.random.normal(size=size_coef))
    else:
        coefficients = initial_coef
    old_score = sampler_score(coefficients)
    if np.isnan(old_score):
        old_score = + 4
    for i in range(100):
        other_random_coefs = np.abs(np.random.normal(size=size_coef))
        new_score = sampler_score(other_random_coefs)
        if new_score < old_score:
            coefficients = other_random_coefs
            old_score = new_score
    if np.isnan(new_score):
        stopHere  # no regions found??
    print('Random initial condition chosen to the best of what random can give us')
    print('Initial score', -sampler_score(coefficients) + 1)
    optimal_coefs = minimize(sampler_score, coefficients, method='nelder-mead')
    print(optimal_coefs.message)
    if optimal_coefs.success is False:
        print('The convergence failed, but the ration between worst region and best region is', -optimal_coefs.fun + 1,
              ', where this is 1 if they have the same number of samples')
    optimal_coef = optimal_coefs.x
    # data = sampler_global(optimal_coef[:n_parameters], optimal_coef[n_parameters:], size_dataset)
    # parameter_region = DSGRN_parameter_region(f, data)
    # np.savez(file_name, optimal_coef=optimal_coef, data=data, parameter_region=parameter_region)
    generate_datafile_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters)
    return file_name


def generate_datafile_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters):
    """
    Takes the optimal coefficients and create a dataset out of them

    INPUT
    file_name       name of output file
    optimal_coef    optimal coefficients for the Fisher distribution
    sampler_global  way to sample from the correct distribution given the optimal parameters
    size_dataset    integer, size of the wanted dataset
    """

    data = sampler_global(optimal_coef[:n_parameters], optimal_coef[n_parameters:], size_dataset)
    parameter_region = assign_region(data)
    np.savez(file_name, optimal_coef=optimal_coef, data=data, parameter_region=parameter_region)
    return file_name


def generate_data_from_coefs(coef, n_parameters, assign_region, size_dataset, sampler=None):
    """
    Takes the optimal coefficients and create a dataset out of them

    INPUT
    optimal_coef    optimal coefficients for the Fisher distribution
    sampler_global  way to sample from the correct distribution given the optimal parameters
    size_dataset    integer, size of the wanted dataset
    """
    if sampler is None:
        sampler = distribution_sampler()
    data = sampler(coef[:n_parameters], coef[n_parameters:], size_dataset)
    parameter_region = assign_region(data)
    return data, parameter_region


def load_dataset(file_name):
    """
    Takes as input the name of the file with a parameter dataset and returns the infomration within

    OUTPUT
    data                parameter values
    parameter_region    number of the parameter region each parameter belongs to
    optimal_coef        coefficients of the appropriate distribution that have been used to create the dataset
    """
    dataset = np.load(file_name)
    return dataset.f.data, dataset.f.parameter_region, dataset.f.optimal_coef


def region_sampler_fisher():
    """
    Creates a sample from the appropriate distribution based on the coefficients given

    Returns a function that takes as input 2 coefficient vectors and the size of the requested sample and that has as
    output a sample
    """

    def fisher_distribution(c1, c2, size):
        return np.random.f(c1, c2, size)

    def many_fisher_distributions(c1_vec, c2_vec, size):
        par = np.zeros([len(c1_vec), size])
        for i in range(len(c1_vec)):
            par[i, :] = fisher_distribution(c1_vec[i], c2_vec[i], size)
        return par

    return many_fisher_distributions


def distribution_sampler():
    """
    Creates a sample from the appropriate normal multivariate distribution based on the coefficients given

    Returns a function that takes as input 2 coefficient vectors and the size of the requested sample and that has as
    output a sample
    """

    def multivariate_normal_distributions(c1_vec, c2_vec, size):
        # par = np.zeros([len(c1_vec), size])
        mean = c1_vec
        dim = len(mean)
        cov = np.reshape(c2_vec, (dim, dim))
        x = np.random.multivariate_normal(mean, cov, size)
        par = np.abs(x).T
        # abs ensures it's positive
        return par

    return multivariate_normal_distributions


def create_dataset_ToggleSwitch(size_dataset, namefile=None, boolAppend=False):
    alpha = np.random.uniform(0, 3, size_dataset)
    beta = np.random.uniform(0, 3, size_dataset)
    parameters = np.array([fiber_sampler(alpha[j], beta[j]) for j in range(size_dataset)])
    parameter_region = associate_parameter_regionTS(alpha, beta)
    if namefile is None:
        namefile = f"ToggleSwitchDataset"
    np.savez(namefile, alpha=alpha, beta=beta, parameters=parameters, parameter_region=parameter_region)
    return


def readTS(file_name=None):
    if file_name is None:
        file_name = f"ToggleSwitchDataset.npz"
    dataset = np.load(file_name)
    return dataset.f.alpha, dataset.f.beta, dataset.f.parameters, dataset.f.parameter_region


def subsample_data_by_region(n_sample, region, alpha, beta, parameters, parameter_region):
    idx = parameter_region.index(region)
    if len(idx) < n_sample:
        raise Exception("Not enough samples to go by")
    sample_idx = idx[random.sample(range(len(idx)), k=n_sample)]
    loc_alpha = alpha[sample_idx]
    loc_beta = beta[sample_idx]
    loc_parameters = parameters[sample_idx, :]
    loc_parameter_region = parameter_region[sample_idx]
    return loc_alpha, loc_beta, loc_parameters, loc_parameter_region


def random_optimize_par(initial_coef, sampler, assign_region, n_pars, iters=100, score=False):
    if not score:
        bin_size = lambda vec: np.array([np.sum(vec == j) for j in range(2)])
        score = lambda vec: min(bin_size(vec)) * len(bin_size(vec)) / np.size(vec)
    best_coef = initial_coef
    best_data = sampler(initial_coef[:n_pars], initial_coef[n_pars:], 500)
    best_parameter_region = assign_region(best_data)
    best_score = score(best_parameter_region)
    for i in range(iters):
        random_coef = best_coef * (1 + np.random.rand(np.size(initial_coef)) * 0.05)
        data = sampler(random_coef[:n_pars], random_coef[n_pars:], 500)
        parameter_region = assign_region(data)
        if score(parameter_region) > best_score:
            best_score = score(parameter_region)
            best_coef = random_coef
    return best_coef, best_score


def subsample_data_by_bounds(n_sample, alpha_min, alpha_max, beta_min, beta_max, alpha, beta, parameters,
                             parameter_region):
    idx = np.nonzero((alpha > alpha_min) * (alpha < alpha_max) * (beta > beta_min) * (beta < beta_max))
    if len(idx) < n_sample:
        raise Exception("Not enough samples to go by")
    sample_idx = idx[random.sample(range(len(idx)), k=n_sample)]
    loc_alpha = alpha[sample_idx]
    loc_beta = beta[sample_idx]
    loc_parameters = parameters[sample_idx, :]
    loc_parameter_region = parameter_region[sample_idx]
    return loc_alpha, loc_beta, loc_parameters, loc_parameter_region


def associate_parameter_regionTS(alpha, beta):
    axes_1 = np.zeros_like(alpha)
    axes_2 = np.zeros_like(alpha)

    axes_1[alpha < 1] = 0
    axes_1[np.logical_and(alpha >= 1, alpha < 2)] = 1
    axes_1[alpha >= 2] = 2

    axes_2[beta < 1] = 0
    axes_2[np.logical_and(beta >= 1, beta < 2)] = 1
    axes_2[beta >= 2] = 2

    matrix_region = axes_1 * 3 + axes_2
    return matrix_region


def DSGRN_parameter_regionTS(parameter):
    # warnings.warn("This function is ONLY CODED FOR THE TOGGLE SWITCH")
    alpha, beta = parameter_to_DSGRN_coord(parameter.T)
    return associate_parameter_regionTS(alpha, beta)


def subsample(file_name, size_subsample):
    data, regions, coefs = load_dataset(file_name)
    size_data = np.size(data, 1)
    if size_subsample > size_data:
        raise ValueError('Cannot ask more samples than the stored ones')
    index_random = np.random.choice(size_data, size=size_subsample, replace=False)
    data_subsample = data[:, index_random]
    region_Subsample = regions[index_random]
    return data_subsample, region_Subsample, coefs


def region_subsample(file_name, region_number, size_subsample):
    data, regions, coefs = load_dataset(file_name)
    subindex_selection, = np.where(regions == region_number)
    data = data[:, subindex_selection]
    size_data = np.size(data, 1)
    if size_subsample > size_data:
        stopHere
    index_random = np.random.randint(0, size_data, size_subsample)
    data_subsample = data[:, index_random]
    return data_subsample, coefs


# let a and b be two vectors in high dimensions, we want to create a distribution that approximately give points along
# the segment [a,b]

def normal_distribution_around_points(a, b):
    v1 = a - b
    lambda_1 = np.linalg.norm(a - b) / 2

    V = np.identity(np.size(a, 1))
    index_info = np.argmax(v1)
    V[:, index_info] = v1
    V[:, [0, index_info]] = V[:, [index_info, 0]]

    Lambda = np.identity(np.size(a, 1))
    Lambda = 10 ** -4 * lambda_1 * Lambda
    Lambda[0, 0] = lambda_1

    V = np.linalg.qr(V.T)[0].T

    Sigma = np.dot(np.dot(V, Lambda), V.T)

    mu = (a[0, :] + b[0, :]) / 2
    return Sigma, mu


def normal_distribution_around_many_points(a, *args):
    size_subspace = len(args)
    central_point = a
    for vec in args:
        central_point = central_point + vec
    mean_point = central_point / (size_subspace + 1)
    average_distance = a - mean_point
    for vec in args:
        average_distance = average_distance + vec - mean_point
    average_distance = average_distance / (size_subspace + 1)

    V = np.identity(np.size(a, 1))
    V[:, 0] = a - mean_point

    Lambda = np.identity(np.size(a, 1))
    lambda_1 = np.linalg.norm(a - mean_point) / 2
    Lambda = 0.0001 * average_distance * Lambda
    Lambda[0, 0] = 0.01 * lambda_1
    i = 1

    for vec in args:
        V[:, i] = vec
        Lambda[i, i] = 0.01 * lambda_1
        i += 1

    V, _ = np.linalg.qr(V.T).T

    Sigma = np.dot(np.dot(V, Lambda), V.T)
    mu = mean_point[0, :]
    return Sigma, mu


# costum specific for Toggle Switch
# create_dataset_ToggleSwitch(10)
# readTS()


def simple_region(x):
    x1 = x[0]
    x2 = x[1]
    assigned_region = np.zeros_like(x1)
    assigned_region[x1 > x2] = 1
    return assigned_region


def second_simple_region(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    assigned_region = np.zeros_like(x1) + 1
    assigned_region[x3 < x1 - x2] = 0
    assigned_region[x3 > x1 + x2] = 2
    return assigned_region


def third_simple_region(x):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    assigned_region1 = np.zeros_like(a) + 1
    assigned_region1[c + d < a * b] = 0
    assigned_region1[a * b < c - d] = 2

    assigned_region2 = np.zeros_like(a)
    assigned_region2[a > b] = 1

    assigned_region = assigned_region1 + 3 * assigned_region2
    return assigned_region


def TS_region(n, name_input):
    n_parameters_TS = 5
    n_regions_TS = 9
    name = create_dataset(n_parameters_TS, DSGRN_parameter_regionTS, n_regions_TS, n, name_input)
    return name


if __name__ == "__main__":
    test_case = np.infty
    if test_case == 0:
        # a < b  ,  a > b
        name = 'simple_test.npz'
        n_parameters_simple = 2
        n_regions_simple = 2
        requested_size = 5000
        name = create_dataset(n_parameters_simple, simple_region, n_regions_simple, requested_size, name)
        data_loc, regions_loc, coefs_optimal = load_dataset(name)
        plt.plot(data_loc[0], data_loc[1], '.')

    if test_case == 1:
        # c < a - b , a-b < c < a+b , a+b < c
        name = 'second_simple_test.npz'
        n_parameters_simple = 3
        n_regions_simple = 3
        requested_size = 5000
        name = create_dataset(n_parameters_simple, second_simple_region, n_regions_simple, requested_size, name)
        data_loc, regions_loc, coefs_optimal = load_dataset(name)
        region_1 = np.sum(data_loc[2, :] < data_loc[0, :] - data_loc[1, :])
        region_3 = np.sum(data_loc[2, :] > data_loc[0, :] + data_loc[1, :])
        region_2 = requested_size - region_1 - region_3

    if test_case == 2:
        # c + d < ab , c-d < ab < c+d , ab < c-d
        # AND a<b, b<a     (6 regions)
        name = 'third_simple_test.npz'
        n_parameters_simple = 4
        n_regions_simple = 6
        requested_size = 5000
        name = create_dataset(n_parameters_simple, third_simple_region, n_regions_simple, requested_size, name)
        data_loc, regions_loc, coefs_optimal = load_dataset(name)
        counter = np.zeros(n_regions_simple)
        for i in range(n_regions_simple):
            counter[i] = np.count_nonzero(regions_loc == i)
        # c < a - b , a-b < c < a+b , a+b < c

    if test_case == 3:
        print('This is the toggle switch')

        # testing region assignment
        # region = associate_parameter_regionTS(np.array([0.5, 0.5, 1.2]), np.array([1.2, 2.4, 0.5]))
        # region should be [1,2,3]

        decay = np.array([1, 1], dtype=float)
        p1 = np.array([1, 5, 3], dtype=float)
        p2 = np.array([1, 6, 3], dtype=float)

        f = ToggleSwitch(decay, [p1, p2])

        name = 'TS_data_test.npz'
        n_parameters_TS = 5
        n_regions_TS = 9
        name = create_dataset(n_parameters_TS, DSGRN_parameter_regionTS, n_regions_TS, 100, name)
        # create a new TS dataset

        testing_functionalities = 0
        if testing_functionalities > 1:
            # expand the dataset (actually, using the same coefs but rewriting the dataset)
            data, parameter_region, coefs_optimal = load_dataset(name)
            sampler_TS = distribution_sampler()
            size_dataset = 100000
            generate_datafile_from_coefs(name, coefs_optimal, sampler_TS, f, size_dataset, n_parameters_TS)

            # subsampling methods: all regions or specific regions
            size_sample = 4
            subsample(name, size_sample)
            region_number = 5
            region_subsample(name, region_number, size_sample)
