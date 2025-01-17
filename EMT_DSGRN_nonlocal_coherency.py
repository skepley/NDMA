import numpy as np
import DSGRN
import random
import matplotlib.pyplot as plt

from hill_model import equilibrium_stability
from models.EMT_model import def_emt_hill_model
from create_dataset import from_region_to_deterministic_point, par_to_region_wrapper, oneregion_dataset, \
    generate_data_from_coefs
from EMT_boxybox import eqs_with_boxyboxEMT
from DSGRNcrawler import DSGRNcrawler

EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)
crawler = DSGRNcrawler(parameter_graph_EMT)
f = def_emt_hill_model()

n_regions = 100
n_parameters_EMT = 42
# hill = 20
hill_vec = [1, 2, 4, 10, 20, 50, 100]
size_dataset_per_region = 20

parameter_regions = random.sample(range(parameter_graph_EMT.size()), n_regions)
_, sources_vec, targets_vec = from_region_to_deterministic_point(EMT_network, 0)


def assign_single_region_fun(region_number):
    assign_region_func = par_to_region_wrapper(f, region_number, parameter_graph_EMT, sources_vec,
                                               targets_vec)
    return assign_region_func


def compute_coherent_percentage(data, hill, DSGRN_n_eq):
    coherent_perc = 0
    for par_NDMA in data:
        NDMA_eqs = eqs_with_boxyboxEMT(hill, par_NDMA)
        NDMA_n_stable_eqs = sum(equilibrium_stability(f, equilibrium, hill, par_NDMA) for equilibrium in NDMA_eqs)
        coherent_perc += (NDMA_n_stable_eqs == DSGRN_n_eq)

    coherent_perc /= np.shape(data)[0]
    return coherent_perc


def data_from_parameter_region(parameter_region):
    assign_region = assign_single_region_fun(parameter_region)

    score, coef = oneregion_dataset(f, parameter_region, size_dataset_per_region, EMT_network, n_parameters_EMT, optimize=False,
                                    save_file=False)
    full_data, assigned_regions = generate_data_from_coefs(coef, n_parameters_EMT, assign_region,
                                                           int(1.2 * size_dataset_per_region / score))
    data = full_data[:, assigned_regions == 0].T
    return data[:size_dataset_per_region, :]


class Result:
    hill = np.array([])
    DSGRN_n_eqs = np.array([])
    coherency_rate = np.array([])

    def update(self, hill, DSGRN_n_eq, coherency_rate):
        self.hill = np.append(self.hill, hill)
        self.DSGRN_n_eqs = np.append(self.DSGRN_n_eqs, DSGRN_n_eq)
        self.coherency_rate = np.append(self.coherency_rate, coherency_rate)
        return self

    def coherency_rate_by_hill(self, hill):
        coherency_vec = np.array([])
        for i in range(np.shape(self.hill)[0]):
            if self.hill[i] == hill:
                coherency_vec = np.append(coherency_vec, self.coherency_rate[i])
        return coherency_vec

    def coherency_rate_by_eqs(self, DSGRN_eqs):
        coherency_vec = np.array([])
        for i in range(np.shape(self.hill)[0]):
            if self.DSGRN_n_eqs[i] == DSGRN_eqs:
                coherency_vec = np.append(coherency_vec, self.coherency_rate[i])
        return coherency_vec

    def coherency_rate_by_eqs_hill(self, DSGRN_eqs, hill):
        coherency_vec = np.array([])
        for i in range(np.shape(self.hill)[0]):
            if self.DSGRN_n_eqs[i] == DSGRN_eqs and self.hill[i] == hill:
                coherency_vec = np.append(coherency_vec, self.coherency_rate[i])
        return coherency_vec


result = Result()
print('Starting multi-region loop\n Completed: 0%')
for parameter_reg in parameter_regions:
    DSGRN_n_eqs = crawler.n_stable_FP(parameter_reg)
    data_in_region = data_from_parameter_region(parameter_reg)
    for hill_val in hill_vec:
        coherent_percentage = compute_coherent_percentage(data_in_region, hill_val, DSGRN_n_eqs)
        result.update(hill_val, DSGRN_n_eqs, coherent_percentage)
    print('\r')
    print('Completed:', np.shape(result.hill)[0]/len(parameter_regions)/len(hill_vec)*100, '%')

print('Taking ', n_regions, ' number of random parameter regions, mean and variance of coherent results is  ',
      np.mean(result.coherency_rate), np.var(result.coherency_rate))

# extract coherency rate VS DSGRN equilibrium number
max_eqs = int(max(result.DSGRN_n_eqs))
n_eqs_mean = np.empty(max_eqs)
n_eqs_var = np.empty(max_eqs)
for n_eqs in range(1, 1+max_eqs):
    coherency = result.coherency_rate_by_eqs_hill(n_eqs, 100)
    n_eqs_mean[n_eqs-1], n_eqs_var[n_eqs-1] = np.mean(coherency), np.var(coherency)

plt.bar(np.array(range(1, 1+max_eqs)), n_eqs_mean)
plt.errorbar(np.array(range(1, 1+max_eqs)), n_eqs_mean, yerr=n_eqs_var, fmt="o", color="r")
plt.xlabel('predicted number of equilibria')
plt.ylabel('mean and variance of coherency rate')
plt.savefig('coherency_rateVSneqs_maxhill.png', bbox_inches='tight')
plt.show()

# coherency rate VS hill
n_eqs_mean = np.empty(len(hill_vec))
n_eqs_var = np.empty(len(hill_vec))
for i in range(len(hill_vec)):
    hill = hill_vec[i]
    coherency = result.coherency_rate_by_hill(hill)
    n_eqs_mean[i], n_eqs_var[i] = np.mean(coherency), np.var(coherency)

plt.bar(np.array(range(len(hill_vec))), n_eqs_mean)
plt.errorbar(np.array(range(len(hill_vec))), n_eqs_mean, yerr=n_eqs_var, fmt="o", color="r")
plt.xticks(np.array(range(len(hill_vec))), hill_vec)
plt.xlabel('hill coefficient')
plt.ylabel('mean and variance of coherency rate')
plt.savefig('coherency_rateVShill.png', bbox_inches='tight')
plt.show()


# coehrency rate VS hill and eqs
max_eqs = int(max(result.DSGRN_n_eqs))
n_eqs_mean = np.empty([max_eqs, len(hill_vec)])
n_eqs_var = np.empty([max_eqs, len(hill_vec)])
for n_eqs in range(1, 1+max_eqs):
    for i in range(len(hill_vec)):
        hill = hill_vec[i]
        coherency = result.coherency_rate_by_eqs_hill(n_eqs, hill)
        n_eqs_mean[n_eqs-1, i], n_eqs_var[n_eqs-1, i] = np.mean(coherency), np.var(coherency)

ax = plt.subplot(111)
bars = []
w = 1/(max_eqs+3)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(len(hill_vec)):
    hill = hill_vec[i]
    bar = ax.bar(np.array(range(1, 1 + max_eqs))+(i-len(hill_vec)/2)*w, n_eqs_mean[:, i], width=w, align='center', color=colors[len(colors)-6-i])
    ax.errorbar(np.array(range(1, 1 + max_eqs))+(i-len(hill_vec)/2)*w, n_eqs_mean[:, i], yerr=n_eqs_var[:, i], fmt="o", color="r")
    #ax.autoscale(tight=True)
    bars.append(bar[0])

plt.xlabel('predicted number of equilibria')
plt.ylabel('mean and variance of coherency rate')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(bars, ['hill = 1', 'hill = 2', 'hill = 4', 'hill = 10', 'hill = 20', 'hill = 50', 'hill = 100'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('coherency_rateVSneqsANDhill.png', bbox_inches='tight')
plt.show()