""" Defines unit tests for the Graph Neural Net (GNN) library. These tests run using the Nose environment.

    After installing Nose, these tests can be run from command line using:
           nosetests -vv test_gnn.py

    After installing Coverage, these tests can report code coverage using:
            nosetests --with-coverage -vv test_gnn.py

    A more practical way to run tests is to simply run:
            nosetests --with-coverage -vv
    This command will run all unit tests in this folder and will report coverage

"""

import unittest
import gnnlib as gnn


class TestGNN(unittest.TestCase):
    def test_initialize_gnn(self):
        st_dim = 4
        lb_dim = 3
        out_dim = 2

        gn = gnn.GraphNet(st_dim, lb_dim, out_dim)

        gn.add_node_list([0, 1, 2, 3, 4, 5, 6])
        gn.add_link_list([(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5), (4, 6), (5, 6), (3, 4)])

        self.assertEqual(gn.get_dim_state(), st_dim, "GNN should report the initialized state dimension")
        self.assertEqual(gn.get_dim_labels(), lb_dim, "GNN should report the initialized label dimension")
        self.assertEqual(gn.get_dim_output(), out_dim, "GNN should report the initialized output vector dimension")

    def test_changing_number_hidden_units(self):
        gn = gnn.GraphNet(4, 3, 2)
        gn.add_node_list([0, 1, 2, 3, 4, 5, 6])
        gn.add_link_list([(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5), (4, 6), (5, 6), (3, 4)])

        gn.new_hidden_vals(15, 20, 12)

        hidden_b, hidden_f, hidden_o = gn.get_hidden_vals()

        self.assertEqual(hidden_b, 15, "GNN should report new value for bias-net hidden dimension")
        self.assertEqual(hidden_f, 20, "GNN should report new value for forcing-net hidden dimension")
        self.assertEqual(hidden_o, 12, "GNN should report new value for output-net hidden dimension")



class TestNeuralNet(unittest.TestCase):
    def test_initialize_nn(self):
        number_inputs = 5
        number_hidden = 6
        number_outputs = 2
        nn = gnn.NeuralNet(number_inputs, number_hidden, number_outputs)

        self.assertEqual(nn.num_inputs, number_inputs, "NN should report the initialized input dimension")
        self.assertEqual(nn.num_hidden, number_hidden, "NN should report the initialized hidden-layer dimension")
        self.assertEqual(nn.num_outputs, number_outputs, "NN should report the initialized output dimension")

    def test_read_initial_coefficients(self):
        number_inputs = 3
        number_hidden = 4
        number_outputs = 2
        nn = gnn.NeuralNet(number_inputs, number_hidden, number_outputs)

        # Number of input coefficient vectors is the same as number of hidden units
        # The size of each input coefficient vector is the same as the input size
        for k in range(4):
            res = nn.get_input_coeffs(k)
            self.assertEqual(max(res), 0, "NN should be initialized with all input coefficients set to zero - max")
            self.assertEqual(min(res), 0, "NN should be initialized with all input coefficients set to zero - min")

        # Number of ouput coefficient vectors is the same as the number of outputs
        # The size of each output coefficient vector is the same as the number of hidden units
        for k in range(2):
            res = nn.get_output_coeffs(k)
            self.assertEqual(max(res), 0, "NN should be initialized with all output coefficients set to zero - max")
            self.assertEqual(min(res), 0, "NN should be initialized with all output coefficients set to zero - min")

    def test_update_input_coefficients(self):
        number_inputs = 3
        number_hidden = 4
        number_outputs = 2
        nn = gnn.NeuralNet(number_inputs, number_hidden, number_outputs)

        nn.update_input_coeffs(0, [10, 11, 12])
        nn.update_input_coeffs(1, [20, 21, 22])
        nn.update_input_coeffs(2, [30, 32, 32])
        # The fourth set has not been updated

        res0 = nn.get_input_coeffs(0)
        res1 = nn.get_input_coeffs(1)
        res2 = nn.get_input_coeffs(2)
        res3 = nn.get_input_coeffs(3)

        self.assertEqual(max(res0), 12, "Largest input coefficient for hidden unit 0 should be 12")
        self.assertEqual(min(res0), 10, "Smallest input coefficient for hidden unit 0 should be 10")

        self.assertEqual(max(res1), 22, "Largest input coefficient for hidden unit 1 should be 22")
        self.assertEqual(min(res1), 20, "Smallest input coefficient for hidden unit 1 should be 20")

        self.assertEqual(max(res2), 32, "Largest input coefficient for hidden unit 2 should be 32")
        self.assertEqual(min(res2), 30, "Smallest input coefficient for hidden unit 02should be 30")

        self.assertEqual(max(res3), 0, "Largest input coefficient for hidden unit 3 should be 0")
        self.assertEqual(min(res3), 0, "Smallest input coefficient for hidden unit 3 should be 0")

    def test_update_output_coefficients(self):
        number_inputs = 3
        number_hidden = 4
        number_outputs = 2
        nn = gnn.NeuralNet(number_inputs, number_hidden, number_outputs)

        nn.update_output_coeffs(0, [50, 51, 52, 53])
        nn.update_output_coeffs(1, [60, 61, 62, 63])

        res0 = nn.get_output_coeffs(0)
        res1 = nn.get_output_coeffs(1)

        self.assertEqual(max(res0), 53, "Largest output coefficient for first output should be 53")
        self.assertEqual(min(res0), 50, "Smallest output coefficient for first output should be 50")

        self.assertEqual(max(res1), 63, "Largest output coefficient for 2nd output should be 63")
        self.assertEqual(min(res1), 60, "Smallest output coefficient for 2nd output should be 60")

    def test_activation_function(self):
        number_inputs = 3
        number_hidden = 4
        number_outputs = 2
        nn1 = gnn.NeuralNet(number_inputs, number_hidden, number_outputs)

        val_1 = nn1.activfun(1)
        val_2 = nn1.activfun(2)
        val_9 = nn1.activfun(9)
        val_m1 = nn1.activfun(-1)
        val_m2 = nn1.activfun(-2)
        val_m9 = nn1.activfun(-9)

        self.assertGreater(val_1, 0.7, "Sigmoid should exceed 0.7 if value is 1")
        self.assertGreater(val_2, 0.85, "Sigmoid should exceed 0.85 if value is 2")
        self.assertGreater(val_9, 0.99, "Sigmoid should exceed 0.99 if value is 9")

        self.assertLess(val_m1, 0.27, "Sigmoid should be lower than 0.27 if value is -1")
        self.assertLess(val_m2, 0.12, "Sigmoid should be lower than 0.12 if value is -2")
        self.assertLess(val_m9, 0.05, "Sigmoid should be lower than 0.05 if value is -9")

    def test_neural_net_compute(self):
        number_inputs = 3
        number_hidden = 4
        number_outputs = 2
        nn = gnn.NeuralNet(number_inputs, number_hidden, number_outputs)

        nn.update_input_coeffs(0, [0, 1, 1])
        nn.update_input_coeffs(1, [1, 0, 0])
        nn.update_input_coeffs(2, [2, 0, 1])
        nn.update_input_coeffs(3, [0, -1, -1])

        nn.update_output_coeffs(0, [0, 1, 0, 1])
        nn.update_output_coeffs(1, [0.5, 0, 0, 2])

        input_vector = [2, 1, 1]

        res = nn.compute_output(input_vector)

        print "nn compute result: ", res

        self.assertGreater(res[0], 0.7, "Computed ouput 0 should exceed 0.72")
        self.assertLess(res[0], 0.75, "Computed output 0 should not exceed 0.74")
        self.assertGreater(res[1], 0.6, "Computated output 1 should exceed 0.66")
        self.assertLess(res[1], 0.7, "Computed output 1 should not exceed 0.0.68")


