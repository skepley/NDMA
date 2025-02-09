"""
Function and design testing for the SaddleNode class

    Output: output
    Other files required: none

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 4/13/2020 
"""
import numpy as np
from ndma.examples.TS_model import ToggleSwitch
from ndma.bifurcation.saddlenode import SaddleNode
import matplotlib.pyplot as plt


def test_saddle_node_search():
    decay = np.array([np.nan, np.nan], dtype=float)
    p1 = np.array([np.nan, np.nan, np.nan], dtype=float)
    p2 = np.array([np.nan, np.nan, np.nan], dtype=float)
    x = np.array([4, 3])

    f = ToggleSwitch(decay, [p1, p2])
    f1 = f.coordinates[0]
    f2 = f.coordinates[1]
    H1 = f1.productionComponents[0]
    H2 = f2.productionComponents[0]
    n0 = 4.1
    par_start = 3.5
    p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3])

    SN = SaddleNode(f)
    eq = f.find_equilibria(3, 10, p0)
    x0 = eq[0, :]

    index_par = 0
    parSol_next_eq = SN.find_saddle_node(index_par, par_start, p0, equilibria=x0)
    parSol = SN.find_saddle_node(index_par, par_start, p0)
    # print(parSol[0], par_start)
    assert np.shape(parSol[0]) == np.shape(par_start)

    # plot nullclines
    plt.figure()
    f.plot_nullcline(4.1, p0)

    plt.figure()
    f.plot_nullcline(parSol, p0)
    assert True