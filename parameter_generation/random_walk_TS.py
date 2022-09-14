import DSGRN
import numpy as np


def random_step(x, step_size = 0.1):
    h = np.random.normal(0, step_size, len(x))
    return x+h


def restricted_random_step(x, bool_region, step_size=0.1):
    h = np.random.normal(0, step_size, len(x))
    iter = 0
    while iter<10 and bool_region(x+h) is False:
        h = np.random.normal(0, step_size, len(x))
        iter = iter+1
        if iter == 10:
            if step_size > 10**-6:
                iter = 0
                step_size = 0.1*step_size
            else:
                AttributeError()
    return x+h

# create network from file
TS_network = DSGRN.Network("TS.txt")
# graph_TS = graphviz.Source(TS_network.graphviz())
# graph_TS.view()
parameter_graph_TS = DSGRN.ParameterGraph(TS_network)

# look into a parameter region
parameterindex = 1
special_parameternode = parameter_graph_TS.parameter(parameterindex)
# print(special_parameternode.inequalities())

# sampling a special parameter node
sampler = DSGRN.ParameterSampler(TS_network)
a = sampler.sample(special_parameternode)
print(a)
