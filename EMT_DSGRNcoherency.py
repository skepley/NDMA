import matplotlib.pyplot as plt
import random
import DSGRN

from ndma.DSGRNintegration.DSGRNcrawler import DSGRNcrawler
from ndma.hill_model import equilibrium_stability
from ndma.basic_models.EMT_model import def_emt_hill_model
from create_dataset import from_region_to_deterministic_point
from EMT_boxybox import eqs_with_boxyboxEMT


EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)
f = def_emt_hill_model()

test_size = 400
hill_vec = [1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 17, 20, 30, 40, 50]

parameter_regions = random.sample(range(parameter_graph_EMT.size()), test_size)
coherency = []

for hill in hill_vec:
    coherent_regions = 0
    for parameter_region in parameter_regions:
        DSGRN_n_eqs = crawler.n_stable_FP(parameter_region)

        par_NDMA, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, parameter_region)
        NDMA_eqs = eqs_with_boxyboxEMT(hill, par_NDMA)
        NDMA_n_stable_eqs = sum(equilibrium_stability(f, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)

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
