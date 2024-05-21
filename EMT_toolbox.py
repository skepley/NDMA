import numpy as np
import scipy
import matplotlib.pyplot as plt
import graphviz
from create_dataset import create_dataset, distribution_sampler, generate_data_from_coefs
import json
from DSGRN_functionalities import *
from models.EMT_model import EMT

# create network from file
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)


def morse_graph_from_index(index):
    """
    takes a parameter index (int) and returns the morse graph from DSGRN
    """
    parameter = parameter_graph_EMT.parameter(index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_graph = DSGRN.MorseGraph(domain_graph)
    return morse_graph


def isFP(morse_node, morse_graph):
    """
    takes a morse node and a morse graph (from DSGRN classes) and returns a boolean if it is a fixed point according to DSGRN
    """
    return morse_graph.annotation(morse_node)[0].startswith('FP')


def is_monostable(par_index):
    """
    takes a parameter index (int) and returns a boolean if the parameter region is monostable
    """
    morse_graph = morse_graph_from_index(par_index)
    return sum(1 for node in range(morse_graph.poset().size()) if isFP(node, morse_graph)) == 1


def vec_is_monostable(vec_index):
    """
    takes a vector of parameter indeces (int) and returns a boolean if the parameter region is monostable
    """
    return [is_monostable(i) for i in vec_index]


def is_bistable(par_index):
    """
    takes a parameter index (int) and returns a boolean if the parameter region is bistable
    """
    morse_graph = morse_graph_from_index(par_index)
    return sum(1 for node in range(morse_graph.poset().size()) if isFP(node, morse_graph)) == 2


def bistable_neighbours(par_index):
    """
    takes a parameter index (int) and returns a list with all the bistable neighbours of the region
    """
    bistable_list = []
    adjacent_nodes = parameter_graph_EMT.adjacencies(par_index)
    for adjacent in adjacent_nodes:
        if is_bistable(adjacent):
            bistable_list.append(adjacent)
    return bistable_list


def def_emt_hill_model():
    """
    returns an instance of the Hill model class describing the EMT model
    """
    # define the EMT hill model
    gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
    edgeCounts = [2, 2, 2, 1, 3, 2]
    parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]  # set all
    # production parameters as variable
    f = EMT(gammaVar, parameterVar)
    return f