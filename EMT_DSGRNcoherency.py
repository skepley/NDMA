from models.EMT_model import *
from create_dataset import *
from EMT_boxybox import eqs_with_boxyboxEMT
from DSGRNcrawler import DSGRNcrawler


EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)
f = def_emt_hill_model()

test_size = 100
coherent_regions = 0
hill = 10

parameter_regions = random.sample(range(parameter_graph_EMT.size()), test_size)

for parameter_region in parameter_regions:
    DSGRN_n_eqs = crawler.n_stable_FP(parameter_region)

    par_NDMA, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, parameter_region)
    NDMA_eqs = eqs_with_boxyboxEMT(hill, par_NDMA)
    NDMA_n_stable_eqs = sum(equilibrium_stability(f, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)

    coherent_regions += (NDMA_n_stable_eqs == DSGRN_n_eqs)

print('Taking ', test_size, ' number of random parameter regions, the ratio of coherent regions is ',
      coherent_regions/test_size)
