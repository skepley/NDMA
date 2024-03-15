"""

"""
import sys

import DSGRN


example_network = DSGRN.Network("example_system.txt")
# graph_EMT = graphviz.Source(EMT_network.graphviz())
# graph_EMT.view()
parameter_graph = DSGRN.ParameterGraph(example_network)

# building the sampler (later used to create a sample parameter per each selected region)
sampler = DSGRN.ParameterSampler(example_network)
num_parameters = parameter_graph.size()
domain_size_EMT = 6

par_index = 51945
print('par_index = ', par_index)
parameter = parameter_graph.parameter(par_index)
domain_graph = DSGRN.DomainGraph(parameter)
morse_graph = DSGRN.MorseGraph(domain_graph)
morse_graph_size = morse_graph.poset().size()
print('poset = ', morse_graph.poset().stringify())
for i in range(morse_graph_size):
    print(morse_graph.annotation(i)[0])

