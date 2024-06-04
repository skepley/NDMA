"""
Tests for the creation of datasets, from the simplest regions to the more complicated ones


Author: Elena Queirolo
Date created: 7 Sept 2021
Date modified:
"""

from create_dataset import *

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
    assigned_region1[c+d < a * b] = 0
    assigned_region1[a * b < c-d] = 2

    assigned_region2 = np.zeros_like(a)
    assigned_region2[a > b] = 1

    assigned_region = assigned_region1 + 3*assigned_region2
    return assigned_region


test_case = 3
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
    region_1 = np.sum(data_loc[2,:] < data_loc[0,:]-data_loc[1,:])
    region_3 = np.sum(data_loc[2,:] > data_loc[0,:]+data_loc[1,:])
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
        sampler_TS = distribution_sampler()
        size_dataset = 100000
        generate_datafile_from_coefs(name, coefs_optimal, sampler_TS, f, size_dataset, n_parameters_TS)

        # subsampling methods: all regions or specific regions
        size_sample = 4
        subsample(name, size_sample)
        region_number = 5
        region_subsample(name, region_number, size_sample)
