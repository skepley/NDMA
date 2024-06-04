import numpy as np
from create_dataset import tworegions_dataset
from models.EMT_model import def_emt_hill_model
from DSGRNcrawler import DSGRNcrawler
import DSGRN

# size and name of dataset created
size_dataset = 10 ** 4
file_name = 'dataset_EMT_may24.npz'
graph_span = 10

print('This code creates a datset of size ', size_dataset, ' in file ', file_name, ' such that ')
print('the file has two information: the data (parameters of EMT) and the DSGRN region they belong to')
print('classified as 0 for monostable, 1 for bistable, 2 for other')
print('The regions are chosen in the first ', graph_span, 'DSGRN regions such that they are adjacent')
print('and each region is maximally enclosed with regions of its same stability')
print('i.e. the monostable region has many monostable regions around,',
      'and the bistable region has many bistable regions around')

# create network from file
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)

possible_regions = np.array(range(graph_span))
monostable_regions = possible_regions[crawler.vec_is_monostable(possible_regions)]
mono_bistable_pairs = []

for par_index_i in monostable_regions:  # parameter_graph_EMT.size()
    bistable_list_i = crawler.bistable_neighbours(par_index_i)
    if bistable_list_i:
        mono_bistable_pairs.append([[par_index_i, bistable_index] for bistable_index in bistable_list_i])

num_parameters = parameter_graph_EMT.size()
num_candidates = len(mono_bistable_pairs)

print('Number of parameters in the parameter graph: ' + str(num_parameters))
print('Monostable parameters found in our search of the parameter graph: ' + str(len(monostable_regions)))
print('Of which, parameters with ad adjacent bistable region: ' + str(num_candidates))
print('All the pairs:\n', mono_bistable_pairs)


def score_many_monostable_and_many_bistable(pair_mono_bi):
    par_index_mono, par_index_bi = pair_mono_bi[0], pair_mono_bi[1]
    adjacent_nodes_mono = parameter_graph_EMT.adjacencies(par_index_mono)
    num_loc_monostable = sum(crawler.vec_is_monostable(adjacent_nodes_mono))
    score_monostable = num_loc_monostable / len(adjacent_nodes_mono)

    adjacent_nodes_bi = parameter_graph_EMT.adjacencies(par_index_bi)
    num_loc_bistable = sum([crawler.is_bistable(adjacent) for adjacent in adjacent_nodes_bi])
    score_bistable = num_loc_bistable / len(adjacent_nodes_bi)

    final_score = score_monostable ** 2 + score_bistable ** 2
    return final_score


# refine the search: to each good candidate count the number of monostable adjacent nodes / number of adjacent nodes and
# the same for the bistable node: we want as many monostable nodes close to the monostable node and as many bistable
# nodes near the bistable node
score_candidate = np.array([score_many_monostable_and_many_bistable(pair[0]) for pair in mono_bistable_pairs])
ranking = score_candidate.argsort()
best_pair = np.array(mono_bistable_pairs[ranking[-1]][0]) # highest score
monostable_region, bistable_region = best_pair[0], best_pair[1]
print('Chosen regions: ' + str(best_pair))

best_pair = np.array([7, 127]) # for testing

f = def_emt_hill_model()
n_parameters_EMT = 42

final_score, _ = tworegions_dataset(f, best_pair, size_dataset, n_parameters_EMT, EMT_network, file_name)
print('Datset created with final score of ', final_score)

