from pylab import imread, gray
import numpy as np
import random

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
        print '%s -> %s receive_msg: %s' % (other.name, self.name, msg)
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        # if as many msgs as neighbours, all other nodes have pending msgs
        if len(self.neighbours) == len(self.in_msgs):
            for node in self.neighbours:
                if node != other:
                    self.pending.add(node)

        # if 1 msg less than amount of neighbours only 1 neighbour has pending msg
        elif len(self.neighbours) == len(self.in_msgs) + 1:
            # find the neighbour
            self.pending.add((set(self.neighbours) - set(self.in_msgs.keys())).pop())

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

        # multiply the in msgs with eachother in order to calculate probability
        marginals = self.observed_state.copy()
        marginals *= np.multiply.reduce(self.in_msgs.values())

        # calculate Z if not provided
        if Z == None:
            Z = np.sum(marginals)

        # normalize
        marginals /= Z

        return marginals, Z

    def max_state(self):
        """ Return the most probable state, assumes max_sum algorithm has run """

        # calculate max sum probability
        max_state = np.log(self.observed_state)
        max_state += np.add.reduce(self.in_msgs.values())

        # return argmax: the maximal state
        return np.argmax(max_state)

    def send_sp_msg(self, other):
        """Send message from Variable to Factor for sum-product algorithm"""
        # check if all necessary msgs are present
        if other in self.pending:
            self.pending.remove(other)
        else:
            raise Exception('%s is not pending' % (other.name,))

        # multiply the incoming msgs with eachother
        messages = [self.in_msgs[node] for node in self.neighbours if node != other]

        out_msg = self.observed_state.copy()
        out_msg *= np.multiply.reduce(messages)

        # send msg
        other.receive_msg(self, out_msg)

    def send_ms_msg(self, other):
        """Send message from Variable to Factor for max-sum algorithm"""
        # check if all necessary msgs are present
        if other in self.pending:
            self.pending.remove(other)
        else:
            raise Exception('%s is not pending' % (other.name,))

        # multiply the incoming msgs with eachother
        messages = [self.in_msgs[node] for node in self.neighbours if node != other]

        out_msg = np.log(self.observed_state)
        out_msg += np.add.reduce(messages)

        # send msg
        other.receive_msg(self, out_msg)

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
        self.log_f = np.log(f)

    def send_sp_msg(self, other):
        """Send message from Factor to Variable for sum-product algorithm"""
        # check if all required information is available
        if other in self.pending:
            self.pending.remove(other)
        else:
            raise Exception('%s is not pending' % (other.name,))

        # compute msg
        messages = [self.in_msgs[node] for node in self.neighbours if node != other]
        messages_prod = np.multiply.reduce(np.ix_(*messages))

        factor_dims = range(self.f.ndim)
        factor_dims.pop(self.neighbours.index(other))
        msg = np.tensordot(messages_prod, self.f, (range(messages_prod.ndim), factor_dims))

        # send msg
        other.receive_msg(self, msg)

    def send_ms_msg(self, other):
        """Send message from Factor to Variable for max-sum algorithm"""
        # check if all required information is available
        if other in self.pending:
            self.pending.remove(other)
        else:
            raise Exception('%s is not pending' % (other.name,))

        # compute msg
        messages = [self.in_msgs[node] if node!= other else np.zeros(other.num_states) for node in self.neighbours]
        messages_add = np.add.reduce(np.ix_(*messages))

        factor_dims = range(self.f.ndim)
        factor_dims.pop(self.neighbours.index(other))
        msg = np.amax(messages_add + self.log_f, tuple(factor_dims))

        # send msg
        other.receive_msg(self, msg)

    def __repr__(self):
        return '%s:\n%s' % (self.name, self.f)

    def __str__(self):
        return self.__repr__()


def sum_product(node_list):
    # initialize pending messages on leave nodes
    for node in node_list:
        if len(node.neighbours) == 1:
            node.pending = set([node.neighbours[0]])
    # send messages to pending nodes
    node_list.extend(reversed(node_list))
    for node in node_list:
        if len(node.pending) == 0:
            print '%s has no pending nodes' % (node.name,)
        while len(node.pending) != 0:
            pending_node = iter(node.pending).next()
            node.send_sp_msg(pending_node)

def max_sum(node_list):
    # initialize pending messages on leave nodes
    for node in node_list:
        if len(node.neighbours) == 1:
            node.pending = set([node.neighbours[0]])
    # send messages to pending nodes
    node_list.extend(reversed(node_list))
    for node in node_list:
        if len(node.pending) == 0:
            print '%s has no pending nodes' % (node.name,)
        while len(node.pending) != 0:
            pending_node = iter(node.pending).next()
            node.send_ms_msg(pending_node)

def loopy_max_sum(nodes, max_iterations):
    """pass messages from randomly chosen nodes iteratively until no more
    pending messages are created or a maximum number of iterations is reached.
    """
    has_pending_nodes = set([])
    # initialize pending messages for leaf nodes
    for node in nodes:
        if len(node.neighbours) == 1:
            node.pending = set([node.neighbours[0]])
            has_pending_nodes.add(node)
    # send messages to pending nodes
    iteration = 0
    while len(has_pending_nodes) > 0 and iteration < max_iterations:
        node = random.sample(has_pending_nodes, 1)[0]
        has_pending_nodes.remove(node)
        while len(node.pending) > 0:
            pending_node = iter(node.pending).next()
            node.send_ms_msg(pending_node)
            if len(pending_node.pending) > 0:
                has_pending_nodes.add(pending_node)

        iteration += 1

    if len(has_pending_nodes) == 0:
        print 'no more pending messages'
    elif iteration >= max_iteratations:
        print 'reached max iterations'

def instantiate1():
    """
    First assignment of notebook, instantiate the network provided
    """

    # holds the nodes in graph
    nodes = dict()

    # append the nodes
    var_names = ['Influenza', 'SoreThroat', 'Fever', 'Bronchitis', \
        'Smokes', 'Wheezing', 'Coughing']
    for var_name in var_names:
        nodes[var_name] = Variable(var_name, 2)

    # append factors

    nodes['priorIN'] = Factor('priorIN', np.array([0.95, 0.05]), [nodes['Influenza']])
    nodes['priorSM'] = Factor('priorSM', np.array([0.8, 0.2]), [nodes['Smokes']])

    nodes['ST-IN'] = Factor('ST-IN', np.array([[0.999, 0.7], [0.001, 0.3]]), [nodes['SoreThroat'], nodes['Influenza']])
    nodes['FE-FL'] = Factor('FE-FL', np.array([[0.95, 0.1], [0.05, 0.9]]), [nodes['Fever'], nodes['Influenza']])
    nodes['CO-BR'] = Factor('CO-BR', np.array([[0.93, 0.2], [0.07, 0.8]]), [nodes['Coughing'], nodes['Bronchitis']])
    nodes['WH-BR'] = Factor('WH-BR', np.array([[0.999, 0.4], [0.001, 0.6]]), [nodes['Wheezing'], nodes['Bronchitis']])


    nodes['BR-IN-SM'] = Factor('BR-IN-SM', np.array([[[0.9999, 0.3], [0.1, 0.01]], [[0.0001, 0.7], [0.9, 0.99]]]), [nodes['Bronchitis'], nodes['Influenza'], nodes['Smokes']])

    nodes['Influenza'].set_observed(1)
    return nodes

def img_to_graph(path):
    """ Convert an img in path to a graph """

    # load img in BW
    im = np.mean(imread(path), axis=2) > 0.5

    # initialize nodes
    x_nodes = [] # 2D for index simplicity
    y_nodes = []
    factors = []

    # the f-matrix of each factor is the same, so defined here
    xy_factor = np.array([[0.5, 0.5],[0.5, 0.5]])
    neighbour_factor = np.array([[0.9, 0.1],[ 0.1, 0.9]])

    # for each row in image
    for i in range(im.shape[0]):

        # holds nodes in this row to be appended to x_nodes
        x_variable_row = []

        # for each individual pixel
        for j in range(im.shape[1]):
            # create nodes for each pixel (latent and observed)
            variableY = Variable("Y_%d_%d" %(i, j), 2)
            variableX = Variable("X_%d_%d" %(i, j), 2)
            # set node to observed
            variableY.set_observed(int(im[i,j]))
            
            # set all factors
            X_Yfactor           = Factor('%s, %s'% (variableY.name,variableX.name), xy_factor , [variableY, variableX]) 
            factors.append(X_Yfactor)
            if i > 0:
                X_Xleft_factor  = Factor('%s, %s'% (x_nodes[i-1][j].name,variableX.name), neighbour_factor , [x_nodes[i-1][j], variableX])
                factors.append(X_Xleft_factor)
                
            if j > 0:
                X_Xup_factor    = Factor('%s, %s'% (x_variable_row[j-1].name,variableX.name), neighbour_factor , [x_variable_row[j-1], variableX])
                factors.append(X_Xup_factor)
                
            # append to lists
            
            x_variable_row.append(variableX)
            y_nodes.append(variableY)            
            
            
        x_nodes.append(x_variable_row)

    x_nodes = np.array(x_nodes).flatten()
    print 'done'
    return (x_nodes, y_nodes, factors)

def get_neighbour_factor(path):
    """ get the correct chances in the factor of an image by counting """

    # load img in BW
    im = np.mean(imread(path), axis=2) > 0.5

    # initialize
    x_diff = 0
    y_diff = 0

    # for each row in image starting with second
    for i in range(1, im.shape[0]):

        # for each individual pixel column starting with second
        for j in range(1, im.shape[1]):
            x_diff += abs(int(im[i-1,j]) - int(im[i,j]))
            y_diff += abs(int(im[i,j-1]) - int(im[i,j]))

    # divide by amount of elements with neighbours
    amount_elements_x = (len(im)-1) * len(im[0])
    amount_elements_y = len(im) * (len(im[0])-1)

    x_diff = x_diff / float(amount_elements_x)
    y_diff = y_diff / float(amount_elements_y)

    print (x_diff, y_diff)

def test_loopy():
    graph = instantiate1()
    nodes = graph.values()
    loopy_max_sum(nodes, len(nodes)*3)
    for node in nodes:
        if isinstance(node, Variable):
            print(str(node).ljust(20) + ' its maximum state: ' + str(node.max_state()))
    img_to_graph('dalmation2.png')


def test_sum_product():
    graph = instantiate1()
    names = ['SoreThroat', 'Fever', 'Coughing', 'Wheezing', 'priorIN',
             'priorSM', 'ST-IN', 'FE-FL', 'CO-BR', 'WH-BR', 'Influenza',
             'Smokes', 'Bronchitis', 'BR-IN-SM']
    nodes = [graph[name] for name in names]
    sum_product(nodes)

    for node in nodes[:len(nodes)/2]:
        if isinstance(node, Variable):
            print(str(node).ljust(15) +  ' has marginal: ' + str(node.marginal()))

def test_max_sum():
    graph = instantiate1()
    names = ['SoreThroat', 'Fever', 'Coughing', 'Wheezing', 'priorIN',
             'priorSM', 'ST-IN', 'FE-FL', 'CO-BR', 'WH-BR', 'Influenza',
             'Smokes', 'Bronchitis', 'BR-IN-SM']
    nodes = [graph[name] for name in names]
    max_sum(nodes)

    for node in nodes[:len(nodes)/2]:
        if isinstance(node, Variable):
            print(str(node).ljust(20) + ' its maximum state: ' + str(node.max_state()))


def test_factor_to_variable_sp():
    graph = instantiate1()
    factor = graph['BR-IN-SM']
    factor.pending = set(factor.neighbours)
    print factor
    messages = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    for i, message in enumerate(messages):
        factor.in_msgs[factor.neighbours[i]] = message

    for other in reversed(factor.neighbours):
        print 'mes towards %s' % other.name
        factor.send_sp_msg(other)
        print other.in_msgs[factor]
        break

def test_variable_to_factor_sp():
    graph = instantiate1()
    variable = graph['Influenza']
    variable.pending = set(variable.neighbours)

    # set msgs in random order
    variable.in_msgs[graph['FE-FL']] = np.array([1, 2])
    variable.in_msgs[graph['BR-IN-SM']] = np.array([3, 4])
    variable.in_msgs[graph['priorIN']] = np.array([5, 6])
    variable.in_msgs[graph['ST-IN']] = np.array([7, 8])

    # send msgs
    for factor in variable.neighbours:
        variable.send_sp_msg(factor)
        print str(factor.name) + ' ' + str(factor.in_msgs[variable])

def test_factor_to_variable_ms():
    graph = instantiate1()
    factor = graph['BR-IN-SM']
    factor.pending = set(factor.neighbours)
    print factor
    messages = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    for i, message in enumerate(messages):
        factor.in_msgs[factor.neighbours[i]] = message

    for other in reversed(factor.neighbours):
        print 'mes towards %s' % other.name
        factor.send_ms_msg(other)
        print other.in_msgs[factor]
        break

def test_variable_to_factor_ms():
    graph = instantiate1()
    variable = graph['Influenza']
    variable.pending = set(variable.neighbours)

    # set msgs in random order
    variable.in_msgs[graph['FE-FL']] = np.array([1, 2])
    variable.in_msgs[graph['BR-IN-SM']] = np.array([3, 4])
    variable.in_msgs[graph['priorIN']] = np.array([5, 6])
    variable.in_msgs[graph['ST-IN']] = np.array([7, 8])

    # send msgs
    for factor in variable.neighbours:
        variable.send_ms_msg(factor)
        print str(factor.name) + ' ' + str(factor.in_msgs[variable])

def test_variable_marginal():
    graph = instantiate1()
    variable = graph['Influenza']

    # set msgs in random order
    variable.in_msgs[graph['FE-FL']] = np.array([1, 2])
    variable.in_msgs[graph['BR-IN-SM']] = np.array([3, 4])
    variable.in_msgs[graph['priorIN']] = np.array([5, 6])
    variable.in_msgs[graph['ST-IN']] = np.array([7, 8])

    # calculate marginal influenza
    marginal, Z = variable.marginal(None)
    print marginal
    print Z

###### Debugging functions ######
def print_graph(graph):
    """
    Print each nodes in the graph
    """

    for node in graph.values():
        print("%s with %d neighbours and of type %s" % \
            (str(node).ljust(20), len(node.neighbours), str(type(node))))

if __name__ == '__main__':
    #graph = instantiate1()
    #print_graph(graph)
    #test_factor_to_variable_sp()
    #test_variable_to_factor_sp()
    #test_factor_to_variable_ms()
    #test_variable_to_factor_ms()
    #test_variable_marginal()
    #test_sum_product()
    #test_max_sum()
    #img_to_graph('../../lab2/dalmatian1.png')
    #img_to_graph('D:\students\Rozeboom\mlpm\mlpm_lab\src\dalmatian1.png')
    test_loopy()
    #get_neighbour_factor('../../lab2/dalmatian1.png')
