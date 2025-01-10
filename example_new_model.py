import numpy as np
from hill_model import HillModel
from models.EMT_model import def_emt_hill_model

# create EMT model
h = def_emt_hill_model()
print('The EMT model: \n', h)

gamma = [np.nan, np.nan, np.nan, np.nan]
p1 = np.array([np.nan, np.nan, np.nan, 3], dtype=float)
p2 = np.array([np.nan, np.nan, np.nan, 3], dtype=float)
p4 = np.array([[np.nan, np.nan, np.nan, 3], [np.nan, np.nan, np.nan, 3], [np.nan, np.nan, np.nan, 3]], dtype=float)
parameter = [p1, p1, p1, p4]

productionSign = [[1], [-1], [1], [1, -1, -1]]
productionType = [[1], [1], [1], [1, 2]]
productionIndex = [[1], [2], [3], [2, 1, 0]]
g = HillModel(gamma, parameter, productionSign, productionType, productionIndex)
print('Example model:\n', g)