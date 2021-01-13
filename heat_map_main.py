from toggle_switch_heat_functionalities import *
from saddle_finding import *

# ============ initialize toggle switch Hill model and SaddleNode instance  ============
# This uses the non-dimensionalized parameters i.e. gamma_1 = 1theta_1 = theta_2 = 1.
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)
hillRange = [1, 150]

sampling_size = 10

monostableParameters = []  # list for indices of monostable parameters
badCandidates = []  # list for parameters which pass the candidate check but fail to find a saddle node
SNParameters = []  # list for parameters where a saddle node is found

parameterData = np.array([sampler() for j in range(sampling_size)])

for j in range(sampling_size):
    parameter = parameterData[j]
    SNParametersj , badCandidatesj = find_saddle_coef(f, hillRange, parameter)
    if SNParametersj == 0 and badCandidatesj == 0:
        monostableParameters.append(j)
    else:
        while SNParametersj:
            SNParameters.append((j, SNParametersj.pop()))

        while badCandidatesj:
            badCandidates.append((j, badCandidatesj.pop()))
    print(j)

dsgrn_plot(SNParameters)
