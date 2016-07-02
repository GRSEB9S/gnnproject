"""
Defines a class and methods to implement Graph Neural Networks (GNN). General forms for GNNs were
studied in [1]. Simplified versions of GNNs have been used to demonstrate their applicability
in different problems [1, 2]. This implementation builds a GNN with the following
elements:

    Each node n has a state x(n) with dimension S (constant across the nodes)
    Each node n has a label vector l(n) with dimension C (constant across the nodes)
    The bias network is a neural network that maps l(n) into b(n)
    The state x(n) is computed using a linear transformation:
        x(n) = sum [ A x(u) ] + b(n)  with a summation over all neighbors of n

    The matrix A is the output of a forcing neural network that takes x(u), l(n), and l(u) as inputs.
    and generates the matrix A as output (of size SxS).

    The node output y(n) can be computed using another neural network (called the output network)
    that uses x(n) and l(n) as inputs. The output dimension is arbitrary and depends on the application

    The GNN input parameters are: S (dim_state), C (dim_labels), and y(n) size (dim_output_vector)

    The GNN class initializes all the required neural networks.

    TBD: The GNN can read a graph collection as training data
    TBD: The GNN can be used to learn particular features of graphs
    TBD: The GNN can be used to predict features or classify new graphs


[1] F. Scarceli et al, "The Graph Neural Network Model", IEEE Transactions on Neural Networks,
    Vol 20, No. 1, 2009
[2] S. R. Baskaraja, M. S. Manickavasagam, "Subgraph Matching Usring Graph Neural Network",
    Journal of Intelligent Learning Systems (online), Novermber 2012
"""
import os
import sys

import math as m
import graphlib as gph
import gnnconst as gct
import responder as resp


__author__ = "Edwin Heredia"
__copyright__ = "Copyright 2016"
__license__ = "Not ready for distribution"
__version__ = "0.0.4"
__status__ = "In development"


class GraphNet(gph.Graph):
    def __init__(self, dim_state, dim_label_vector, dim_output_vector):
        gph.Graph.__init__(self)
        self.__sdim = dim_state          # dimension of any state vector
        self.__ldim = dim_label_vector   # dimension of any label vector
        self.__odim = dim_output_vector  # dimension of any output vector
        self.__bnets = {}        # stores all bias networks indexed by node_id
        self.__fnets = {}        # stores all forcing networks indexed by node_id. Each node has several forcing nets
        self.__onets = {}        # stores all output networks indexed by node_id
        self.__statevars = {}    # stores state variables and outputs indexed by node_id

        # default values for number of hidden units - can be changed by using the new_hidden_vals function
        self.__bnet_hidden = 10
        self.__fnet_hidden = 10
        self.__onet_hidden = 10

    def get_dim_state(self):
        """
        Retrieves the size of node state variables
        Returns:
            An integer number representing the dimension of any node state variable
        """
        return self.__sdim

    def get_dim_labels(self):
        """
        Retrieves the size of node labels
        Returns:
            An integer number representing the dimension of any node label
        """
        return self.__ldim

    def get_dim_output(self):
        """
        Retrieves the size of the output vector
        Returns:
            An integer number representing the dimensions of the output vector
        """
        return self.__odim

    def new_hidden_vals(self, b_hidden=None, f_hidden=None, o_hidden=None):
        """
        Changes the number of hidden units for the three types of neural networks in a GNN
        Args:
            b_hidden: Number of hidden units for a bias neural net
            f_hidden: Number of hidden units for a forcing neural net
            o_hidden: Number of hidden units for an output neural net

        """
        if b_hidden is not None:
            self.__bnet_hidden = b_hidden
        if f_hidden is not None:
            self.__fnet_hidden = f_hidden
        if o_hidden is not None:
            self.__onet_hidden = o_hidden

    def get_hidden_vals(self):
        """
        Retrieve the current dimensions for the hidden layers in the 3 types of neural nets in a GNN
        Returns:
            The dimension of hidden layers for "bias net", "forcing nets", and "output net"
        """
        return self.__bnet_hidden, self.__fnet_hidden, self.__onet_hidden

    def create_nets(self, node_id):
        """
        After creating a graph node, if the node carries a GNN then this method must be called
        to initialize the GNN. This method initializes the data structures that will host
        neural networks, state variables, labels, and output.
        Args:
            node_id: (integer or string) Identifies the node for which a GNN is defined
        """
        if gph.Graph.node_exists(self, node_id):
            self.__add_bias_net(node_id)
            self.__add_forcing_nets(node_id)
            self.__add_output_net(node_id)
            self.__statevars[node_id] = {"state": self.__sdim*[0],   # state initialized with zeros
                                         "labels": self.__ldim*[0],  # labels initialized with zeros
                                         "output": None}             # no ouput has been generated yet

    def __add_bias_net(self, node_id):
        """
        Add a bias network to the node identified by node_id
        This neural network maps l(n) into b(n). The dimension of l(n) is ldim while the
        dimension of b(n) is sdim.

        Args:
            node_id: (integer or string) The ID for the target node
        """
        # initialize a neural network
        neural_net = NeuralNet(self.__ldim, self.__bnet_hidden, self.__sdim)

        # store a reference to the neural network in the bnets dictionary
        if gph.Graph.node_exists(self, node_id):
            if node_id not in self.__bnets.keys():
                self.__bnets[node_id] = neural_net

    def __add_forcing_nets(self, node_id):
        """
        Add several forcing networks to the node identified by node_id (one per neighbor)
        A forcing neural network uses x(u), l(u), and l(n) as inputs (input size 2*C + S) and
        generates matrix A as output. The matrix A has a dimension of SxS. The output from
        the network corresponds to the matrix as a column vector.

        Args:
            node_id: (integer or string) The ID for the target node
        """
        if gph.Graph.node_exists(self, node_id):
            nbors = gph.Graph.get_neighbors(self, node_id)

            for contrib in nbors:
                self.__add_single_forcing_net(node_id, contrib)

    def __add_single_forcing_net(self, node_id, contrib_id):
        """
        Add a forcing network to the node identified by node ID using node with contrib_id
        as the contributor node.
        Args:
            node_id: (integer or string) The target node for the forcing neural net
            contrib_id: (integer or string) The contributing node for this forcing neural net
        """
        # the input vector consists of x(u), l(u), and l(n). The output is a matrix of size S*S, which
        # for the purpose of the network, is represented as a vector
        input_vec_size = 2*self.__ldim + self.__sdim
        output_vec_size = self.__sdim * self.__sdim
        neural_net = NeuralNet(input_vec_size, self.__fnet_hidden, output_vec_size)

        if gph.Graph.node_exists(self, node_id) and gph.Graph.node_exists(self, contrib_id):
            if node_id not in self.__fnets.keys():
                if contrib_id not in self.__fnets[node_id].keys():
                    self.__fnets[node_id][contrib_id] = neural_net

    def __add_output_net(self, node_id):
        """
        Add an output network to the node identified by node_id
        This neural network maps x(n) and l(n) into b(n). The dimension of l(n) is ldim,
        the dimension of b(n) is sdim, and the dimension of the output is odim

        Args:
            node_id: (integer or string) The ID for the target node
        """
        # initialize a neural network
        neural_net = NeuralNet(self.__sdim + self.__ldim, self.__onet_hidden, self.__odim)

        # store a reference to the neural network in the onets dictionary
        if gph.Graph.node_exists(self, node_id):
            if node_id not in self.__onets.keys():
                self.__onets[node_id] = neural_net

    def has_bias_net(self, node_id):
        """
        Checks if a bias network has been defined for the node identified by node_id
        Args:
            node_id: (integer or string) The target node
        Returns:
            True if a bias network has been defined.
        """
        return node_id in self.__bnets.keys()

    def has_forcing_nets(self, node_id):
        """
        Checks if at least one forcing network has been defined for the node identified by node_id
        Args:
            node_id: (integer or string) The target node
        Returns:
            True if at least one forcing network has been defined.
        """
        return node_id in self.__fnets.keys()

    def has_forcing_net_link(self, node_id, contrib_id):
        """
        Checks if a forcing network has been defined for node_id with contribution from contrib_id
        Args:
            node_id: (integer or string) The target node
            contrib_id: (integer or string) The contributing node
        Returns:
            True if a forcing network with contribution from contrib_id has been defined for node_id
        """
        if node_id in self.__fnets.keys():
            if contrib_id in self.__fnets[node_id].keys():
                return True
            else:
                return False
        else:
            return False

    def compute_output(self, graph_sample):
        # todo: implement compute_output function
        pass

    def train(self, output_matrix, graph_matrix):
        # todo: implement a GNN training algorithm
        pass


class NeuralNet(object):
    """
    Defines a generic 3-layer neural-network for use in GNNs

    The input parameters are: number of inputs (num_inputs), number of hidden nodes (num_hidden),
    the number of outputs (num_outputs), and the activation function, which by default is
    set to "sigmoid".
    """
    def __init__(self, num_inputs=0, num_hidden=0, num_outputs=0, activation="sigmoid"):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.activation = activation

        # initialize coefficient matrices with zeros
        # a matrix is stored as a list of lists (one list per node). There are two sets of coefficients:
        # The ones that apply weights to the input (inpcoeff) and the ones that apply weights to generate
        # the output (outcoeff)
        zvec1 = [0]*num_inputs
        self.inpcoeff = [zvec1]*num_hidden

        zvec2 = [0]*num_hidden
        self.outcoeff = [zvec2]*num_outputs

    def is_initialized(self):
        """
        Checks to see if the number of inputs, hidden units, and outputs have been defined
        Returns:
            True if they have been initialized to some non-zero value. Otherwise it returns False.
        """
        if self.num_inputs == 0 or self.num_hidden == 0 or self.num_outputs == 0:
            return False
        else:
            return True

    def initialize(self, num_inputs, num_hidden, num_outputs, activation="sigmoid"):
        """
        Initializes the vector dimensions that define a neural network. These values are often entered at the
        time of instantiating a neural network object. These values can also be retrieved from a file, in which
        case this function is used programatically to initialize values.
        Args:
            num_inputs: (int) number of inputs in a neural network
            num_hidden: (int) number of hidden units in a neural network
            num_outputs: (int) number of outputs in a neural network
            activation: (string) identifier for the activation function
        Returns:
            If successful it returns 1. Otherwise it returns 0.
        """

        if num_inputs <= 0 or num_hidden <= 0 or num_outputs <= 0:
            return 0
        elif not self.is_activation(activation):
            return 0
        else:
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.num_hidden = num_hidden
            self.activation = activation

            # initialize coefficient matrices with zeros
            # a matrix is stored as a list of lists (one list per node). There are two sets of coefficients:
            # The ones that apply weights to the input (inpcoeff) and the ones that apply weights to generate
            # the output (outcoeff)
            zvec1 = [0] * num_inputs
            self.inpcoeff = [zvec1] * num_hidden

            zvec2 = [0] * num_hidden
            self.outcoeff = [zvec2] * num_outputs

            return 1

    def get_input_coeffs(self, hidden_index):
        """
        Retrieve the input coefficients that correspond to a particular hidden node.
        Args:
            hidden_index: An integer between 0 and (num_hidden_nodes - 1)
        Returns:
            The coefficients that generate the input for the hidden node defined by hidden_index
        """
        if 0 <= hidden_index < self.num_hidden:
            return self.inpcoeff[hidden_index]
        else:
            return None

    def get_output_coeffs(self, output_index):
        """
        Retrieve the output coefficients that correspond to a particular output.
        Args:
            output_index: An integer between 0 and (num_of_outputs - 1)
        Returns:
            The coefficients that generate the input for the output identified by output_index
        """
        if 0 <= output_index < self.num_outputs:
            return self.outcoeff[output_index]
        else:
            return None

    def update_input_coeffs(self, hidden_index, coeff_vector):
        """
        Update the input coefficients that correspond to a particular hidden node
        Args:
            hidden_index: An integer between 0 and (num_hidden_nodes - 1)
            coeff_vector: The coefficient vector that will replace the existing vector
        Returns:
            The number of updated vectors
        """
        if 0 <= hidden_index < self.num_hidden and len(coeff_vector) == self.num_inputs:
            self.inpcoeff[hidden_index] = coeff_vector
            return 1    # indicates that one vector was updated
        else:
            return 0    # indicates that no vector has been updated

    def update_output_coeffs(self, output_index, coeff_vector):
        """
        Update the output coefficients that generate a particular output
        Args:
            output_index: An integer between 0 and (num_of_outputs - 1)
            coeff_vector: The coefficient vector that will replace the existing vector
        Returns:
            The number of updated vectors
        """
        if 0 <= output_index < self.num_outputs and len(coeff_vector) == self.num_hidden:
            self.outcoeff[output_index] = coeff_vector
            return 1    # indicates that one vector was updated
        else:
            return 0    # indicates that no vector has been updated

    def compute_output(self, input_vector):
        """
        Given an input vector, this function computes the output using the neural network
        Args:
            input_vector: The input vector of size num_inputs
        Returns:
             The output vector of size num_outputs
        """
        if len(input_vector) != self.num_inputs:
            return None

        # compute the output of all hidden layer units
        hvec = []
        for cvec in self.inpcoeff:
            combination_out = sum([x*w for x, w in zip(input_vector, cvec)])
            hvec.append(self.activfun(combination_out))

        # compute output vector
        yvec = []
        for cvec in self.outcoeff:
            combination_out = sum([x*w for x, w in zip(hvec, cvec)])
            yvec.append(self.activfun(combination_out))

        return yvec

    def compute_datafile(self):
        """
        Computes the output of a neural network for data stored in a file. The function
        reads the file from stdin and writes the result to stdout

        Returns:
            An outcome message with a simple OK or an error message
        """

        for line in sys.stdin:
            raw_values = line.strip().split()
            if len(raw_values) == self.num_inputs:
                real_values = [float(val) for val in raw_values]
                outvals = self.compute_output(real_values)
                if outvals:
                    strvals = [str(val) for val in outvals]
                    print "    ".join(strvals)
                else:
                    print "error"
            else:
                print "error"

        return resp.ok("na", "na")

    def activfun(self, value):
        """
        Defines activation functions for the network
        Args:
            value: input value
        Returns:
            The output value that results from transforming the input value
        """
        if self.activation == 'sigmoid':
            output = 1.0 / (1.0 + m.exp(-1.0*value))
        else:
            output = 1 if value > 0 else 0
        return output

    def is_activation(self, function_name):
        """
        Checks if the function name corresponds to an activation function
        Arges:
            function_name: (string) User-entered activation function
        Returns:
            True if the function_name is one of the available activation functions.
            False otherwise.
        """
        if function_name == 'sigmoid':
            return True
        else:
            return False

    def train(self, output_matrix, input_matrix):
        # todo: implement a NN training algorithm
        pass

    def save(self, filepath):
        """
            Saves the neural network model to a file identified by its path. The file is a text file.

            Args:
                filepath: An absolute or relative path
            Returns:
                An outcome message with a simple OK or with an error message
        """

        if self.num_inputs == 0 or self.num_hidden == 0 or self.num_outputs == 0:
            return resp.error("nn_init")

        try:
            with open(filepath, "w") as fp:
                numinp_line = "inputs" + "\t" + str(self.num_inputs) + "\n"
                numhid_line = "hidden" + "\t" + str(self.num_hidden) + "\n"
                numout_line = "outputs" + "\t" + str(self.num_outputs) + "\n"
                activ_line = "activation" + "\t" + self.activation + "\n"

                fp.write(numinp_line)
                fp.write(numhid_line)
                fp.write(numout_line)
                fp.write(activ_line)

                for ind in range(self.num_hidden):
                    cvec = self.get_input_coeffs(ind)
                    cstr = [str(val) for val in cvec]
                    cline = "\t".join(cstr) + "\n"
                    fp.write(cline)

                for ind in range(self.num_outputs):
                    kvec = self.get_output_coeffs(ind)
                    kstr = [str(val) for val in kvec]
                    kline = "\t".join(kstr) + "\n"
                    fp.write(kline)

            return resp.ok("na", "na")

        except IOError:
            return resp.error("file_open")

    def read(self, filepath):
        """
        Reads neural network coefficients from a file defined by its path
        Args:
            filepath: (string) The file path with neural network coefficients
        Return:
            An outcome message with a simple OK or with an error message
        """
        # TODO: Check early termination of read loops and return an error?

        if not os.path.isfile(filepath):
            return resp.error("not_found")

        error_found = False
        with open(filepath, "r") as fp:
            line = fp.readline()
            inp = int(line.strip().split("\t")[1])
            line = fp.readline()
            hid = int(line.strip().split("\t")[1])
            line = fp.readline()
            out = int(line.strip().split("\t")[1])

            line = fp.readline()
            activ = line.strip().split("\t")[1]

            if inp <= 0 or hid <= 0 or out <= 0:
                error_found = True

            elif inp > gct.MAX_INPUTS or hid > gct.MAX_HIDDEN or out > gct.MAX_OUTPUTS:
                error_found = True

            elif not self.is_activation(activ):
                error_found = True

            else:
                self.initialize(inp, hid, out, activ)

                for ind in range(hid):
                    line = fp.readline()
                    cvals = line.strip().split("\t")
                    coeff = [float(val) for val in cvals]

                    if len(coeff) != inp:
                        error_found = True
                        break
                    else:
                        self.update_input_coeffs(ind, coeff)

                if not error_found:
                    for ind in range(out):
                        line = fp.readline()
                        cvals = line.strip().split("\t")
                        coeff = [float(val) for val in cvals]

                        if len(coeff) != hid:
                            error_found = True
                            break
                        else:
                            self.update_output_coeffs(ind, coeff)

            if error_found:
                return resp.error("data_parse")
            else:
                return resp.ok("na", "na")


if __name__ == "__main__":
    number_inputs = 3
    number_hidden = 4
    number_outputs = 2
    nn = NeuralNet(number_inputs, number_hidden, number_outputs)

    nn.update_input_coeffs(0, [0, 1, 1])
    nn.update_input_coeffs(1, [1, 0, 0])
    nn.update_input_coeffs(2, [2, 0, 1])
    nn.update_input_coeffs(3, [0, -1, -1])

    nn.update_output_coeffs(0, [0, 1, 0, 1])
    nn.update_output_coeffs(1, [0.5, 0, 0, 2])

    print "saving created NN into a file called nncreated.nnf"
    res = nn.save("nncreated.nnf")

    if res["status"] == "error":
        print res["message"]
        os.sys.exit(1)

    n2 = NeuralNet()
    print "reading NN from file nncreated.nnf"
    res = n2.read("nncreated.nnf")

    if res["status"] == "error":
        print res["message"]
        os.sys.exit(1)

    print "result of reading: ", res["status"]

    print "saving imported NN into a file called nntest.nnf"
    n2.save("nntest.nnf")
