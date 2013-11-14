import numpy as np

class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
         
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
        
    def reset(self):
        # Incomming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    
    def receive_msg(self, other, msg):
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        # TODO: add pending messages
        # self.pending.update(...)
    
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate observed an latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        
    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: Z. Either equal to the input Z, or computed (if Z=None was passed).
        """
        # TODO: compute marginal
        return None, None
    
    def send_sp_msg(self, other):
        # TODO: implement Variable -> Factor message for sum-product
        pass
   
    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f
        
    def send_sp_msg(self, other):
        # TODO: implement Factor -> Variable message for sum-product

        # check if all required information is available

        # compute msg

        # send msg
   
    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass

def instantiate1():
    """
    First assignment of notebook, instantiate the network provided
    """

    # holds the nodes in graph
    nodes = dict()

    # append the nodes 
    varNames = ['Influenza', 'SoreThroat', 'Fever', 'Bronchitis', \
        'Smokes', 'Wheezing', 'Coughing']
    for varName in varNames:
      nodes[varName] = Variable(varName, 2)

    # append factors

    nodes['priorIN'] = Factor('priorIN', np.array([0.95, 0.05]), [nodes['Influenza']])
    nodes['priorSM'] = Factor('priorSM', np.array([0.8, 0.2]), [nodes['Smokes']])

    nodes['ST-IN'] = Factor('ST-IN', np.array([[0.009, 0.7], [0.001, 0.3]]), [nodes['SoreThroat'], nodes['Influenza']])
    nodes['FE-FL'] = Factor('FE-FL', np.array([[0.95, 0.1], [0.05, 0.9]]), [nodes['Fever'], nodes['Influenza']])
    nodes['CO-BR'] = Factor('CO-BR', np.array([[0.03, 0.2], [0.07, 0.8]]), [nodes['Coughing'], nodes['Bronchitis']])
    nodes['WH-BR'] = Factor('WH-BR', np.array([[0.009, 0.4], [0.001, 0.6]]), [nodes['Wheezing'], nodes['Bronchitis']])


    nodes['BR-IN-SM'] = Factor('BR-IN-SM', np.array([[[0.0009, 0.3], [0.9, 0.99]], [[0.0001, 0.7], [0.1, 0.01]]]), [nodes['Bronchitis'], nodes['Influenza'], nodes['Smokes']])


    return nodes

###### Debugging functions ######
def print_graph(graph):
    """
    Print each nodes in the graph
    """

    for node in graph.values():
        print("%s with %d neighbours and of type %s" % \
            (str(node).ljust(20), len(node.neighbours), str(type(node))))

if __name__ == '__main__':
    graph = instantiate1()
    print_graph(graph)
