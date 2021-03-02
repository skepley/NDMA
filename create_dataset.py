"""
Some code to create and manage a huge data set of stored parameters, created a priori, then accessed as needed


Author: Elena Queirolo
Created: 1st March 2021
Modified: 1st March 2021
"""
from hill_model import *
import numpy as np
from toggle_switch_heat_functionalities import *
import random
from scipy.optimize import minimize
from datetime import datetime
import warnings


def create_dataset(f: HillModel, n_parameter_region: int, size_dataset: int, file_name=None, boolAppend=False):
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        file_name = f"{timestamp}"
    sampler_global = region_sampler()

    def sampler_score(fisher_coefficients):

        data_sample = sampler_global(fisher_coefficients[:n_parameters], fisher_coefficients[n_parameters:], 10**3)
        data_region = DSGRN_parameter_region(f, data_sample)
        # TODO: link to DSGRN, this takes as input a matrix of parameters par[1:n_pars,1:size_sample], and returns a
        # vector data_region[1:size_sample], such that daa_region[i] tells us which region par[:, i] belongs to
        # data_region goes from 0 to n_parameter_region -1
        counter = np.zeros(n_parameter_region)
        for i in range(n_parameter_region):
            counter[i] = np.count_nonzero(data_region == i)
        score = 1 - np.min(counter)/np.max(counter)
        # print(score) # lowest score is best score!
        return score  # score must be minimized

    n_parameters = 5 # f.nVariableParameter
    coefficients = np.abs(np.random.normal(size=2*n_parameters)/5)
    for i in range(100):
        other_random_coefs = np.abs(np.random.normal(size=2*n_parameters)/5)
        if sampler_score(other_random_coefs) < sampler_score(coefficients):
            coefficients = other_random_coefs
    print('Random initial condition chosen to the best of what random can give us')
    optimal_coefs = minimize(sampler_score, coefficients, method='nelder-mead')
    print(optimal_coefs.message)
    if optimal_coefs.success is False:
        print('The convergence failed, but the ration between worst region and best region is', optimal_coefs.fun,
              ', where this is 1 if they have the same number of samples')
    optimal_coef = optimal_coefs.x
    data = sampler_global(optimal_coef[:n_parameters], optimal_coef[n_parameters:], size_dataset)
    parameter_region = DSGRN_parameter_region(f, data)
    np.savez(file_name, data=data, parameter_region=parameter_region)
    return file_name


def load_dataset(file_name):
    dataset = np.load(file_name)
    return dataset.data, dataset.parameter_region


def region_sampler():
    def fisher_distribution(c1, c2, size):
        return np.random.f(c1, c2, size)

    def many_fisher_distributions(c1_vec, c2_vec, size):
        par = np.zeros([len(c1_vec), size])
        for i in range(len(c1_vec)):
            par[i, :] = fisher_distribution(c1_vec[i], c2_vec[i], size)
        return par
    return many_fisher_distributions


def create_dataset_ToggleSwitch(size_dataset, namefile=None, boolAppend=False):
    alpha = np.random.uniform(0, 3, size_dataset)
    beta = np.random.uniform(0, 3, size_dataset)
    parameters = np.array([fiber_sampler(alpha[j], beta[j]) for j in range(size_dataset)])
    parameter_region = associate_parameter_regionTS(alpha, beta)
    if namefile is None:
        namefile = f"ToggleSwitchDataset"
    with open(f"{namefile}", "w") as output_file:
        output_file.write(write(alpha, beta, parameters, parameter_region))


def readTS(namefile=None):
    if namefile is None:
        namefile = f"ToggleSwitchDataset"
    with open(f"{namefile}") as input_file:
        text = input_file.read()
    lines = text.split("\n")
    n_sample = int(lines[0])
    alpha = np.empty(shape=n_sample)
    beta = np.empty(shape=n_sample)
    parameter =  np.empty(shape=[n_sample, 5])
    parameter_region = np.empty(shape=n_sample)
    for i, line in enumerate(lines[1:n_sample+1]):
        [alpha_loc, beta_loc, _, p1, p2, p3, p4, p5, _, parameter_region_loc] = line.split()
        alpha[i] = float(alpha_loc)
        beta[i] = float(beta_loc)
        parameter[i, 0] = float(p1)
        parameter[i, 1] = float(p2)
        parameter[i, 2] = float(p3)
        parameter[i, 3] = float(p4)
        parameter[i, 4] = float(p5)
        parameter_region[i] = float(parameter_region_loc)
    return alpha, beta, parameter, parameter_region


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


def subsample_data_by_bounds(n_sample, alpha_min, alpha_max, beta_min, beta_max, alpha, beta, parameters, parameter_region):
    idx = np.nonzero((alpha > alpha_min) * (alpha < alpha_max)*(beta > beta_min) * (beta < beta_max))
    if len(idx) < n_sample:
        raise Exception("Not enough samples to go by")
    sample_idx = idx[random.sample(range(len(idx)), k=n_sample)]
    loc_alpha = alpha[sample_idx]
    loc_beta = beta[sample_idx]
    loc_parameters = parameters[sample_idx, :]
    loc_parameter_region = parameter_region[sample_idx]
    return loc_alpha, loc_beta, loc_parameters, loc_parameter_region


def write(alpha, beta, parameters, parameter_region) -> str:
    output_lines = [f"{len(alpha)}"]
    for i in range(len(alpha)):
        output_lines.append(f"{alpha[i]} {beta[i]} [ {parameters[i,0]} {parameters[i,1]} {parameters[i,2]} {parameters[i,3]} {parameters[i,4]} ] {parameter_region[i]}")
    return "\n".join(output_lines)


def associate_parameter_regionTS(alpha, beta):
    if is_vector(alpha):
        parameter_region = np.zeros_like(alpha)
        for j in range(len(parameter_region)):
            parameter_region[j] = associate_parameter_regionTS(alpha[j], beta[j])
        return parameter_region
    if alpha < 2:
        if alpha < 1:
            axes_1 = 0
        else:
            axes_1 = 1
    else:
        axes_1 = 2
    if beta < 2:
        if beta < 1:
            axes_2 = 0
        else:
            axes_2 = 1
    else:
        axes_2 = 2
    matrix_region = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return matrix_region[axes_1, axes_2]


def DSGRN_parameter_region(_, parameter):
    warnings.warn("This function is ONLY CODED FOR THE TOGGLE SWITCH")
    alpha, beta = parameter_to_DSGRN_coord(parameter.T)
    return associate_parameter_regionTS(alpha, beta) - 1


create_dataset_ToggleSwitch(10)
readTS()
create_dataset(None, 9, 100) # create a new TS dataset