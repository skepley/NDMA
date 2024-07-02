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


np.seterr(over='ignore', invalid='ignore')  # ignore overflow and division by zero warnings:


EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)
f = def_emt_hill_model()

n_parameters_EMT = 42
hill = 20
size_dataset = 200
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


coherent_percentage_monostable = coherent_percentage(data_in_region_monostable[:size_dataset, :], 1)
coherent_percentage_bistable = coherent_percentage(data_in_region_bistable[:size_dataset, :], 2)

print('Taking ', size_dataset, ' number of random parameter regions, at hill coefficient ', hill, 'in the monostable \n',
      'region we have a coherency rate of ', coherent_percentage_monostable, ' and in the bistable ',
      coherent_percentage_bistable)


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


saddle_perc_monostable = prevalence_vertical_saddles(data_in_region_monostable)
saddle_perc_bistable = prevalence_vertical_saddles(data_in_region_bistable)

print("In the monostable regions, data generated a percentage of ", saddle_perc_monostable, ' saddles')
print("In the bistable regions, data generated a percentage of ", saddle_perc_bistable, ' saddles')


# find the single parameter of difference between the two regions
inequalities_monostable = parameter_graph_EMT.parameter(selected_pair[0]).inequalities()
inequalities_bistable = parameter_graph_EMT.parameter(selected_pair[1]).inequalities()

for i in range(len(inequalities_monostable)):
    if inequalities_monostable[i] != inequalities_bistable[i]:
        print(inequalities_monostable[i - 3:i + 60])
        print(inequalities_bistable[i - 3:i + 100])
        break

print('We can transition from region A to region B by increasing T[X4->X3] in DSGRN coordinates')
print('This corresponds to theta_{3,0} where the 3 comes from X3 and the 0 comes from it being \n',
      'the only term in the equation for X3')


def theta_bound(fixed_pars):
    # T[X4->X3] in region B is bounded by U[X1->X4] L[X2->X4] L[X3->X4]
    location_ells_4theqs_fixed_pars = 5 + 7 * 3 + np.array([0, 3, 6])
    ells = fixed_pars[location_ells_4theqs_fixed_pars]
    delta = fixed_pars[location_ells_4theqs_fixed_pars[0] + 1]
    bound = (ells[0]+delta)*ells[1]*ells[2]
    return bound


def search_horizontal_saddles(par42, hill):
    f = def_emt_hill_model()
    saddle_node_problem = SaddleNode(f)
    parameter_index_no_hill = 4 + 6 * 3 + 2
    parameter_index_with_hill = parameter_index_no_hill + 1
    min_theta = par42[parameter_index_no_hill]
    theta_selection = np.linspace(min_theta, theta_bound(par42), 200)
    par_with_hill = np.append(hill, par42)
    SNParameters, badCandidates = saddle_node_with_boxybox_THETA(saddle_node_problem, theta_selection, par_with_hill,
                                                                 parameter_index_with_hill)
    if SNParameters and SNParameters != 0:
        return True
    else:
        return False


n_saddles = 0
for data in data_in_region_monostable[:size_dataset, :]:
    saddle_horizontal = search_horizontal_saddles(data, hill)
    n_saddles += saddle_horizontal
print(n_saddles, 'out of ', size_dataset)
