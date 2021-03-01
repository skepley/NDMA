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


def create_dataset_ToggleSwitch(size_dataset, namefile=None, boolAppend=False):
    alpha = np.random.uniform(0, 3, size_dataset)
    beta = np.random.uniform(0, 3, size_dataset)
    parameters = np.array([fiber_sampler(alpha[j], beta[j]) for j in range(size_dataset)])
    parameter_region = associate_parameter_regionTS(alpha, beta)
    if namefile is None:
        namefile = f"ToggleSwitchDataset"
    with open(f"{namefile}", "w") as output_file:
        output_file.write(write(alpha, beta, parameters, parameter_region))


def redTS(namefile=None):
    if namefile is None:
        namefile = f"ToggleSwitchDataset"
    with open(f"{namefile}") as input_file:
        text = input_file.read()
    lines = text.split("\n")
    [alpha, beta, parameters, parameter_region] = [int(i) for i in lines[0].split()]
    # ask Ruben about this!
    return alpha, beta, parameters, parameter_region


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
        output_lines.append(f"{alpha[i]} {beta[i]} {parameters[i,:]} {parameter_region[i]}")
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


create_dataset_ToggleSwitch(10)
