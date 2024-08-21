"""
This code creates the heat map, as presented in the article.
The heat map indicates the value of the Hill coefficient in which a saddle node is taking place depending on the parameter.
It also consider the parameter projection into [0,3]x[0,3] thanks to the DSGRN region definition
"""

from hill_model import *
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import numpy as np
import matplotlib.pyplot as plt
from models.TS_model import ToggleSwitch
import sys
from create_dataset import *

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# use dataset creation
# size of the sample
n_sample = 10 ** 3  # testing on 3, final run on 4
file_name = 'TS_data_100000.npz'
try:
    np.load(file_name)
except FileNotFoundError:
    n = 100000
    create_dataset_ToggleSwitch(100000, file_name)

data_subsample, region_subsample, coefs = subsample(file_name, n_sample)
a = np.transpose(data_subsample)

hill_vec = [1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 17, 20, 30, 40, 50, 60, 70, 80, 90, 100]
gridDensity = 5
coherency_rate, coherency_rate_R5, coherency_rate_donut = np.empty(0), np.empty(0), np.empty(0)
for hill in hill_vec:
    n_5 = 0
    n_donut = 0
    coherency = 0
    coherency_R5 = 0
    coherency_donut = 0
    for j in range(n_sample):
        a_j = a[j, :]
        region = region_subsample[j]
        n_eqs = count_eq(f, hill, a_j)
        # n_eqs = np.shape(eqs)[0]
        if region == 4:
            n_5 += 1
            if n_eqs == 3:
                coherency += 1
                coherency_R5 += 1
        else:
            n_donut += 1
            if n_eqs == 1:
                coherency += 1
                coherency_donut += 1
    coherency_rate = np.append(coherency_rate, coherency/n_sample)
    coherency_rate_R5 = np.append(coherency_rate_R5, coherency_R5/n_5)
    coherency_rate_donut = np.append(coherency_rate_donut, coherency_donut/n_donut)

plt.figure()
plt.ylim(0, 1.01)
plt.plot(hill_vec, coherency_rate, label='all parameters')
plt.plot(hill_vec, coherency_rate_R5, label='bistable region')
plt.plot(hill_vec, coherency_rate_donut, label='monostable region')
plt.xlabel('Hill coefficient')
plt.ylabel('coherency rate')
plt.legend()
plt.savefig('coherency_VS_hill_TS.pdf')
plt.show()



