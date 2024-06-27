import numpy as np
from hill_model import equilibrium_stability
from models.EMT_model import def_emt_hill_model
from create_dataset import from_region_to_deterministic_point, par_to_region_wrapper, \
    generate_data_from_coefs, tworegions_dataset
from EMT_boxybox import eqs_with_boxyboxEMT, saddle_node_with_boxybox, saddle_node_with_boxybox_THETA
from DSGRNcrawler import DSGRNcrawler
import DSGRN
from saddle_node import SaddleNode
from hill_model import HillModel

EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)
f = def_emt_hill_model()

test_size = 20
n_parameters_EMT = 42
hill = 20
size_dataset = 20
optimize_bool = False

selected_pair = [14643097617, 14643061617]
print('Chosen regions: ' + str(selected_pair))

_, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, 0)
assign_region = par_to_region_wrapper(f, selected_pair, parameter_graph_EMT, sources_vec,
                                      targets_vec)

score, coef = tworegions_dataset(f, selected_pair, size_dataset, EMT_network, n_parameters_EMT, save_file=False,
                                 optimize=optimize_bool)

data, assigned_regions = generate_data_from_coefs(coef, n_parameters_EMT, assign_region,
                                                  int(2.4 * size_dataset / score))
data_in_region_monostable = data[:, assigned_regions == 0].T
data_in_region_bistable = data[:, assigned_regions == 1].T


def coherent_percentage(data_vec, expected_eqs):
    coherent_perc = 0
    for par_NDMA in data_vec:
        NDMA_eqs = eqs_with_boxyboxEMT(hill, par_NDMA)
        NDMA_n_stable_eqs = sum(equilibrium_stability(f, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)
        coherent_perc += (NDMA_n_stable_eqs == expected_eqs)

    coherent_perc /= np.shape(data_vec)[0]
    return coherent_perc


# coherent_percentage_monostable = coherent_percentage(data_in_region_monostable[:size_dataset, :], 1)
# coherent_percentage_bistable = coherent_percentage(data_in_region_bistable[:size_dataset, :], 2)

# print('Taking ', test_size, ' number of random parameter regions, at hill coefficient ', hill, 'in the monostable \n',
#      'region we have a coherency rate of ', coherent_percentage_monostable, ' and in the bistable ',
#      coherent_percentage_bistable)


def prevalence_vertical_saddles(data):
    saddle_node_problem = SaddleNode(f)
    hill_selection = [1, 2, 3, 4, 5, 7, 9, 10, 15, 20, 40, 70, 100]
    saddle_occurences = 0
    for par in data:
        SNParameters, badCandidates = saddle_node_with_boxybox(saddle_node_problem, hill_selection, par)
        if SNParameters and SNParameters != 0:
            saddle_occurences += 1
    saddle_likelyhood = saddle_occurences / np.shape(data)[0]
    return saddle_likelyhood

"""
saddle_perc_monostable = prevalence_vertical_saddles(data_in_region_monostable)
saddle_perc_bistable = prevalence_vertical_saddles(data_in_region_bistable)

print("In the monostable regions, data generated a percentage of ", saddle_perc_monostable, ' saddles')
print("In the bistable regions, data generated a percentage of ", saddle_perc_bistable, ' saddles')
"""
