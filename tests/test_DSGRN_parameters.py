import json

import DSGRN
import numpy as np

from ndma.model.model import Model
import DSGRN_functionalities
from ndma.model.restricted_model import HillModelRestricted

net_spec = """x : x + y
            y : y (~x)"""

network = DSGRN.Network(net_spec)
parameter_graph = DSGRN.ParameterGraph(network)

par_index = 0
parameter = parameter_graph.parameter(par_index)

A = Model.Model_from_string(net_spec)
A_restricted = HillModelRestricted.Model_from_Model(A)
print('Full model :\n', A)
print('Restricted model :\n', A_restricted)

# instead, do this, already all coded
pars, indices_sources, indices_targets = DSGRN_functionalities.from_region_to_deterministic_point(network, par_index)
hill = 3.2
print('Evaluating the restricted model at the given parameters')
x = np.array([2.3,5.4])
print(A_restricted(x, hill, pars))