# goal of this piece of code: compute the coherency rate w.r.t. the hill coefficient in the now-renamed BR (Bernardo
# Rivas) system
# NETWORK_SPEC = """x : x + y
#                   y : (~x) y"""

# we have also access to BR's parameter data (that should be likely translated into NDMA data format)

import numpy as np
import matplotlib.pyplot as plt
import DSGRN
import random

from ndma.DSGRNintegration.DSGRNcrawler import DSGRNcrawler
from ndma.boxy_box import boxy_box, equilibria_with_boxybox
from ndma.hill_model import equilibrium_stability
from ndma.DSGRNintegration.DSGRN_functionalities import from_region_to_deterministic_point
from ndma.model import Model, RestrictedHillModel

string = """x : x + y\n y : y(~x) """
model = Model.Model_from_string(string)
BR_model = RestrictedHillModel.Model_from_Model(model)

print(BR_model)

example_parameter = np.abs(np.random.randn(15))

test = BR_model(np.array([2,.3]), example_parameter)
print('Running the BR model suceeded: ', test)

x_low, x_high = boxy_box(BR_model, example_parameter)
print('the boxy box returns ', x_low, x_high)

EMT_network = DSGRN.Network(string)
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)

test_size = 400
hill_vec = [1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 17, 20, 30, 40, 50]

parameter_regions = random.sample(range(parameter_graph_EMT.size()), test_size)
coherency = []

for hill in hill_vec:
    coherent_regions = 0
    for parameter_region in parameter_regions:
        DSGRN_n_eqs = crawler.n_stable_FP(parameter_region)

        par_NDMA, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, parameter_region)
        NDMA_eqs = equilibria_with_boxybox(hill, par_NDMA)
        NDMA_n_stable_eqs = sum(equilibrium_stability(BR_model, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)

        coherent_regions += (NDMA_n_stable_eqs == DSGRN_n_eqs)
    coherency.append(coherent_regions/test_size)
    print(f'Taking  {test_size} and hill coefficient ',
          f'{hill} number of random parameter regions, '
          f'the ratio of coherent regions is {coherent_regions/test_size}')

plt.figure()
plt.ylim(0, 1.01)
plt.plot(hill_vec, coherency)
plt.xlabel('Hill coefficient')
plt.ylabel('coherency rate')
plt.savefig('coherency_VS_hill.pdf')
plt.show()



