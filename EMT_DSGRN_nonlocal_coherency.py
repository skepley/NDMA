import numpy as np

from models.EMT_model import *
from create_dataset import *
from EMT_boxybox import eqs_with_boxyboxEMT
from DSGRNcrawler import DSGRNcrawler

EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)
f = def_emt_hill_model()

test_size = 20
n_parameters_EMT = 42
hill = 20
size_dataset = 20

parameter_regions = random.sample(range(parameter_graph_EMT.size()), test_size)
_, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, 0)


def assign_single_region_fun(region_number):
    assign_region_func = par_to_region_wrapper(f, region_number, parameter_graph_EMT, sources_vec,
                                          targets_vec)
    return assign_region_func


vec_coeherent_percentages = np.array([])
for parameter_region in parameter_regions:
    DSGRN_n_eqs = crawler.n_stable_FP(parameter_region)

    assign_region = assign_single_region_fun(parameter_region)

    score, coef = oneregion_dataset(f, parameter_region, size_dataset, EMT_network, n_parameters_EMT, optimize=False, save_file=False)
    data, assigned_regions = generate_data_from_coefs(coef, n_parameters_EMT, assign_region, int(1.2*size_dataset/score))
    data_in_region = data[:, assigned_regions == 0].T

    coherent_percentage = 0
    for par_NDMA in data_in_region:
        NDMA_eqs = eqs_with_boxyboxEMT(hill, par_NDMA)
        NDMA_n_stable_eqs = sum(equilibrium_stability(f, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)
        coherent_percentage += (NDMA_n_stable_eqs == DSGRN_n_eqs)

    coherent_percentage /= np.shape(data_in_region)[0]

    vec_coeherent_percentages = np.append(vec_coeherent_percentages, coherent_percentage)

print('Taking ', test_size, ' number of random parameter regions, mean and variance of coherent results is  ',
      np.mean(vec_coeherent_percentages), np.var(vec_coeherent_percentages))
