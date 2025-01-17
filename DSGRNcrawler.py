import DSGRN


def isFP(morse_node, morse_graph):
    """
    takes a morse node and a morse graph (from DSGRN classes) and returns a boolean if it is a fixed point according to DSGRN
    """
    return morse_graph.annotation(morse_node)[0].startswith('FP')


# create network from file
EMT_network = DSGRN.Network("EMT.txt")
parameter_graph_EMT = DSGRN.ParameterGraph(EMT_network)


class DSGRNcrawler:
    """
    This class allows you to create an instance that knows how to crawl the given DSGRN parameter graph. In particular,
    the instance has the following functionalities:
     - morse_graph_from_index = determine the morse graph of a given parameter region (define as an integer)
     - isFP = determine if the given node in the griven morse graph is a fixed point
     - is_monostable = determines is a given parameter region (given as an integer) is monostable
     - vec_is_monostable = determines if the list of parameter regions (given as an iterable of integers) is elementwise
            monostable
     - is_bistable = determines is a given parameter region (given as an integer) is bistable
     - vec_is_bistable = determines if the list of parameter regions (given as an iterable of integers) is elementwise
            bistable
     - bistable_neighbours = determines which neighbours of the given parameter region (given as an integer) are bistable
    """

    def __init__(self, parameter_graph=parameter_graph_EMT):
        self.parameter_graph = parameter_graph

    def morse_graph_from_index(self, index):
        """
        takes a parameter index (int) and returns the morse graph from DSGRN
        """
        parameter = self.parameter_graph.parameter(index)
        domain_graph = DSGRN.DomainGraph(parameter)
        morse_graph = DSGRN.MorseGraph(domain_graph)
        return morse_graph

    def is_monostable(self, par_index):
        """
        takes a parameter index (int) and returns a boolean if the parameter region is monostable
        """
        morse_graph = self.morse_graph_from_index(par_index)
        return sum(1 for node in range(morse_graph.poset().size()) if isFP(node, morse_graph)) == 1

    def vec_is_monostable(self, vec_index):
        """
        takes a vector of parameter indeces (int) and returns a boolean if the parameter region is monostable
        """
        return [self.is_monostable(i) for i in vec_index]

    def is_bistable(self, par_index):
        """
        takes a parameter index (int) and returns a boolean if the parameter region is bistable
        """
        morse_graph = self.morse_graph_from_index(par_index)
        return sum(1 for node in range(morse_graph.poset().size()) if isFP(node, morse_graph)) == 2

    def vec_is_bistable(self, vec_index):
        """
        takes a vector of parameter indeces (int) and returns a boolean if the parameter region is monostable
        """
        return [self.is_bistable(i) for i in vec_index]

    def bistable_neighbours(self, par_index):
        """
        takes a parameter index (int) and returns a list with all the bistable neighbours of the region
        """
        bistable_list = []
        adjacent_nodes = parameter_graph_EMT.adjacencies(par_index)
        for adjacent in adjacent_nodes:
            if self.is_bistable(adjacent):
                bistable_list.append(adjacent)
        return bistable_list

    def n_stable_FP(self, par_index):
        """
        takes a parameter index (int) and returns a boolean if the parameter region is bistable
        """
        morse_graph = self.morse_graph_from_index(par_index)
        return sum(1 for node in range(morse_graph.poset().size()) if isFP(node, morse_graph))

