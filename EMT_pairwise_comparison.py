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

print('We can transition from region A to region B by increasing T[X3->X4] in DSGRN coordinates')
print('This corresponds to theta_{4,2} where the 4 comes from X4 and the 2 comes from the existence of \n',
      'T[X1->X4], T[X2->X4] and T[X3->X4]')

productionSign = [[-1, -1], [-1, -1], [1, -1], [-1], [-1, 1, -1],
                  [-1, -1]]  # length 6 list of production signs for each node
productionType = [len(sign) * [1] for sign in productionSign]  # all productions are products
productionIndex = [[0, 1, 3], [1, 2, 4], [2, 0, 5], [3, 4], [4, 1, 2, 3], [5, 2, 4]]
edgeCounts_EMT = [2, 2, 2, 1, 3, 2]
# before theta_{4,2} there are 4 gamma values, ell, delta, and 10 full terms (each having 4 variables)
# this is  a hacky way to find the index of the tenth theta
parameters_EMT = ['g', 'l', 'd', 't', 'h', 'l', 'd', 't', 'h',
                  'g', 'l', 'd', 't', 'h', 'l', 'd', 't', 'h',
                  'g', 'l', 'd', 't', 'h', 'l', 'd', 't', 'h',
                  'g', 'l', 'd', 't', 'h',
                  'g', 'l', 'd', 't', 'h', 'l', 'd', 't', 'h', 'l', 'd', 't', 'h',
                  'g', 'l', 'd', 't', 'h', 'l', 'd', 't', 'h']


def position_search(what, how_many):
    location = np.array([], int)
    start_index_search = 0
    for i in range(how_many):
        location = np.append(location, parameters_EMT.index(what, start_index_search))
        start_index_search = int(location[-1])+1
    return location


location_tenth_theta = position_search('t', 10)[-1]
location_all_hill = position_search('h', 12)
location_non_hill = np.array(list(set.difference(set(range(54)), set(location_all_hill))))
location_all_gamma = position_search('g', 6)
location_ells_4theqs = position_search('l', 10)[7:]
location_important_delta = location_ells_4theqs[0] + 1


def reshape_pars(fixed_pars_48):
    reshaped_array = np.reshape(fixed_pars_48, (12, 4))
    structured_par = [reshaped_array[0:2, :], reshaped_array[2:4, :], reshaped_array[4:6, :], reshaped_array[6, :],
                      reshaped_array[7:10, :], reshaped_array[10:12, :]]
    return structured_par


def def_EMT_fixed_pars(fixed_pars):
    parameter_54 = np.empty(54)
    parameter_54[location_non_hill] = fixed_pars
    parameter_54[location_all_hill] = 20
    parameter_54[location_tenth_theta] = np.nan
    gamma = parameter_54[location_all_gamma]
    parameter_no_gamma = np.delete(parameter_54, location_all_gamma)
    EMT_not_hill = HillModel(gamma, reshape_pars(parameter_no_gamma), productionSign, productionType, productionIndex)
    return EMT_not_hill, parameter_54  # gamma, parameter_no_gamma


def theta_bound(fixed_pars):
    location_ells_4theqs_fixed_pars = np.array(location_ells_4theqs) - np.array([7, 8, 9])
    ells = fixed_pars[location_ells_4theqs_fixed_pars]
    delta = fixed_pars[location_important_delta - 7]
    bound = (ells[0]+delta)*ells[1]*ells[2]
    return bound


def is_theta_bounded(fixed_pars):
    bound = theta_bound(fixed_pars)
    theta = fixed_pars[location_tenth_theta - 7]
    return theta < bound


def is_in_monostable_reg(fixed_pars):
    location_ells_4theqs_fixed_pars = np.array(location_ells_4theqs) - np.array([7, 8, 9])
    ells = fixed_pars[location_ells_4theqs_fixed_pars]
    bound = ells[0]*ells[1]*ells[2]
    theta = fixed_pars[location_tenth_theta - 7]
    return theta < bound


def is_in_bistable_reg(fixed_pars):
    if is_in_monostable_reg(fixed_pars):
        return False
    return is_theta_bounded(fixed_pars)


def search_horizontal_saddles(par):
    f, parameter54 = def_EMT_fixed_pars(par)
    saddle_node_problem = SaddleNode(f)
    min_theta = par[location_tenth_theta - 9]
    theta_selection = np.linspace(min_theta, theta_bound(par), 3)
    SNParameters, badCandidates = saddle_node_with_boxybox_THETA(saddle_node_problem, theta_selection, parameter54)
    if SNParameters and SNParameters != 0:
        return True
    else:
        return False


n_saddles = 0
for data in data_in_region_monostable[:size_dataset, :]:
    saddle_horizontal = search_horizontal_saddles(data)
    n_saddles += saddle_horizontal
print(n_saddles)
