"""
Some code to create and manage a huge data set of stored parameters, created a priori, then accessed as needed


Author: Elena Queirolo
Created: 1st March 2021
Modified: 1st March 2021
"""
from toggle_switch_heat_functionalities import *
import random
from scipy.optimize import minimize
from datetime import datetime
from ndma.basic_models.TS_model import ToggleSwitch
import matplotlib.pyplot as plt


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
    region_sampler
    DSGRN_parameter_region
    generate_data_from_coefs
    """
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        file_name = f"{timestamp}" + '.npz'

    sampler_global = region_sampler()
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
    generate_data_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters)
    return file_name


def generate_data_from_coefs(file_name, optimal_coef, sampler_global, assign_region, size_dataset, n_parameters):
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


def multivariate_normal_distributions(c1_vec, c2_vec, size):
    # par = np.zeros([len(c1_vec), size])
    mean = c1_vec
    dim = len(mean)
    cov = np.reshape(c2_vec, (dim, dim))
    x = np.random.multivariate_normal(mean, cov, size)
    par = np.abs(x).T
    # abs ensures it's positive
    return par


def region_sampler():
    """
    Creates a sample from the appropriate normal multivariate distribution based on the coefficients given

    Returns a function that takes as input 2 coefficient vectors and the size of the requested sample and that has as
    output a sample
    """

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


def subsample(file_name, size_subsample, wanted_regions=[]):
    def flatten(xss):
        return [x for xs in xss for x in xs]

    data, regions, coefs = load_dataset(file_name)
    if wanted_regions:
        indices = flatten([np.where(regions == wanted_regions[i])[0] for i in range(len(wanted_regions))])
        regions = regions[indices]
        data = data[:, indices]
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
        sampler_TS = region_sampler()
        size_dataset = 100000
        generate_data_from_coefs(name, coefs_optimal, sampler_TS, f, size_dataset, n_parameters_TS)

        # subsampling methods: all regions or specific regions
        size_sample = 4
        subsample(name, size_sample)
        region_number = 5
        region_subsample(name, region_number, size_sample)
