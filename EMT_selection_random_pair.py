import numpy as np
import DSGRN
import random

from ndma.hill_model import equilibrium_stability
from ndma.examples.EMT_model import def_emt_hill_model
from create_dataset import from_region_to_deterministic_point, par_to_region_wrapper, \
    generate_data_from_coefs, tworegions_dataset
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
graph_span = 10
optimize_bool = False  # for the dataset creation

print('This code creates a datset of size ', size_dataset, ' such that ')
print('the file has two information: the data (parameters of EMT) and the DSGRN region they belong to')
print('classified as 0 for monostable, 1 for bistable, 2 for other')
print('The regions are chosen in the first ', graph_span, 'DSGRN regions such that they are adjacent')
print('and each region is maximally enclosed with regions of its same stability')
print('i.e. the monostable region has many monostable regions around,',
      'and the bistable region has many bistable regions around')
score = 0
while score < 0.2:
    possible_regions = np.array(random.sample(range(parameter_graph_EMT.size()), graph_span))
    monostable_regions = possible_regions[crawler.vec_is_monostable(possible_regions)]
    mono_bistable_pairs = []

    for par_index_i in monostable_regions:  # parameter_graph_EMT.size()
        bistable_list_i = crawler.bistable_neighbours(par_index_i)
        if bistable_list_i:
            mono_bistable_pairs.append([[par_index_i, bistable_index] for bistable_index in bistable_list_i])

    num_candidates = len(mono_bistable_pairs)
    if num_candidates < 1:
        continue

    random_pair = np.array(mono_bistable_pairs[0][0])
    monostable_region, bistable_region = random_pair[0], random_pair[1]

    print('Chosen regions: ' + str(random_pair))

    _, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, 0)
    assign_region = par_to_region_wrapper(f, random_pair, parameter_graph_EMT, sources_vec,
                                              targets_vec)

    score, coef = tworegions_dataset(f, random_pair, size_dataset, EMT_network, n_parameters_EMT, save_file=False,
                                     optimize=optimize_bool)

data, assigned_regions = generate_data_from_coefs(coef, n_parameters_EMT, assign_region,
                                                      int(1.2 * size_dataset / score))
data_in_region_monostable = data[:, assigned_regions == 0].T
data_in_region_bistable = data[:, assigned_regions == 1].T


def coherent_percentage(data, expected_eqs):
    coherent_percentage = 0
    for par_NDMA in data:
        NDMA_eqs = eqs_with_boxyboxEMT(hill, par_NDMA)
        NDMA_n_stable_eqs = sum(equilibrium_stability(f, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)
        coherent_percentage += (NDMA_n_stable_eqs == expected_eqs)

    coherent_percentage /= np.shape(data)[0]
    return coherent_percentage


coherent_percentage_monostable = coherent_percentage(data_in_region_monostable, 1)
coherent_percentage_bistable = coherent_percentage(data_in_region_bistable, 2)

print('Taking ', test_size, ' number of random parameter regions, at hill coefficient ', hill, 'in the monostable \n',
      'region we have a coherency rate of ', coherent_percentage_monostable, ' and in the bistable ',
      coherent_percentage_bistable)

