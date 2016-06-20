""" Defines unit tests for the Graph library. These tests run using the Nose environment.

    After installing Nose, these tests can be run from command line using:
           nosetests -vv test_graph.py

    After installing Coverage, these tests can report code coverage using:
            nosetests --with-coverage -vv test_graph.py

    A more practical way to run tests is to simply access the folder for programs and run:
            nosetests --with-coverage -vv
    This command will run all unit tests in this folder and will report coverage

"""

import unittest
import graphlib as grl


class TestGraph(unittest.TestCase):

    def test_create_graph(self):
        my_graph = grl.Graph()
        self.assertEquals(my_graph.size(), 0, "Graph should return a size 0 upon initialization")

    def test_graph_size(self):
        gr = grl.Graph()

        for k in range(4):
            gr.add_node(k)

        self.assertEquals(gr.size(), 4, "Graph should have a size of 4")
        gr.add_node(10)

        gr.add_node(11)
        self.assertEquals(gr.size(), 6, "Graph should have a size of 6")

    def test_node_existence(self):
        gr = grl.Graph()

        for k in range(4):
            gr.add_node(k)

        self.assertTrue(gr.node_exists(0), "Graph should report that node 0 exists")
        self.assertTrue(gr.node_exists(1), "Graph should report that node 1 exists")
        self.assertTrue(gr.node_exists(2), "Graph should report that node 2 exists")
        self.assertTrue(gr.node_exists(3), "Graph should report that node 3 exists")

        self.assertFalse(gr.node_exists(10), "Graph should report that node 10 does not exist")
        self.assertFalse(gr.node_exists(20), "Graph should report that node 20 does not exist")

    def test_node_list(self):
        gr = grl.Graph()

        for k in range(5):
            gr.add_node(k)

        gr.add_link(0, 1)
        gr.add_link(0, 2)
        gr.add_link(2, 3)
        gr.add_link(1, 3)
        gr.add_link(1, 4)
        gr.add_link(3, 4)
        gr.add_link(2, 4)

        res = gr.get_nodes()

        self.assertIn(0, res, "Node list should include Node 0")
        self.assertIn(1, res, "Node list should include Node 1")
        self.assertIn(2, res, "Node list should include Node 2")
        self.assertIn(3, res, "Node list should include Node 3")
        self.assertIn(4, res, "Node list should include Node 4")

    def test_add_links(self):
        gr = grl.Graph()

        for k in range(5):
            gr.add_node(k)

        gr.add_link(0, 1)
        gr.add_link(0, 2)
        gr.add_link(2, 3)
        gr.add_link(1, 3)
        gr.add_link(1, 4)
        gr.add_link(3, 4)
        gr.add_link(2, 4)

        res1 = set(gr.get_links(0))
        self.assertEquals(len(res1), 2, "Node 0 shoudl report two links")

        self.assertIn(1, res1, "Node 1 should be listed as a neighbor of Node 0")
        self.assertIn(2, res1, "Node 2 should be listed as a neighbor of Node 0")

        res2 = set(gr.get_links(4))
        self.assertEquals(len(res2), 3, "Node 4 shoudl report three links")

        self.assertIn(1, res2, "Node 1 should be listed as a neighbor of Node 4")
        self.assertIn(2, res2, "Node 2 should be listed as a neighbor of Node 4")
        self.assertIn(3, res2, "Node 3 should be listed as a neighbor of Node 4")

    def test_remove_links(self):
        gr = grl.Graph()

        for k in range(5):
            gr.add_node(k)

        gr.add_link(0, 1)
        gr.add_link(0, 2)
        gr.add_link(2, 3)
        gr.add_link(1, 3)
        gr.add_link(1, 4)
        gr.add_link(3, 4)
        gr.add_link(2, 4)

        self.assertEquals(gr.size(), 5, "Graph should report 5 nodes")

        self.assertIn(2, set(gr.get_links(4)), "Node 2 shoudl be listed as neighbor of 4")
        self.assertIn(4, set(gr.get_links(2)), "Node 4 should be listed as neighbor of 2")

        gr.remove_link(2, 4)

        self.assertNotIn(2, set(gr.get_links(4)), "Node 2 should not be a neighbor of 4")
        self.assertNotIn(4, set(gr.get_links(2)), "Node 4 should not be a neighbor of 2")

    def test_remove_node(self):
        gr = grl.Graph()

        for k in range(5):
            gr.add_node(k)

        gr.add_link(0, 1)
        gr.add_link(0, 2)
        gr.add_link(2, 3)
        gr.add_link(1, 3)
        gr.add_link(1, 4)
        gr.add_link(3, 4)

        self.assertEquals(gr.size(), 5, "Graph should report 5 nodes")
        self.assertIn(1, set(gr.get_links(0)), "Node 1 should be listed as neighbor of 0")
        self.assertIn(0, set(gr.get_links(1)), "Node 0 should be listed as neighbor of 1")
        self.assertIn(2, set(gr.get_links(0)), "Node 2 should be listed as neighbor of 0")
        self.assertIn(0, set(gr.get_links(2)), "Node 0 should be listed as neighbor of 2")

        gr.remove_node(0)

        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")
        self.assertFalse(gr.node_exists(0), "Graph should report that node 0 does not exist")

        self.assertNotIn(0, set(gr.get_links(1)), "Node 0 should NOT be listed as neighbor of 1")
        self.assertNotIn(0, set(gr.get_links(2)), "Node 0 should NOT be listed as neighbor of 2")

    def test_add_and_remove_nodes_by_list(self):
        gr = grl.Graph()

        gr.add_node_list(['a', 'b', 'c', 'd'])
        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")

        gr.remove_node_list(['b', 'd'])
        self.assertEquals(gr.size(), 2, "Graph should report 2 nodes")

    def test_add_and_remove_links_by_list(self):
        gr = grl.Graph()

        gr.add_node_list(['a', 'b', 'c', 'd'])
        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")

        list1 = [('a', 'b'), ('b', 'c'), ('a', 'd'), ('c', 'd')]
        gr.add_link_list(list1)

        self.assertTrue(gr.link_exists('a', 'b'), "Graph should include a link between 'a' and 'b'")
        self.assertTrue(gr.link_exists('a', 'd'), "Graph should include a link between 'a' and 'd'")

        list2 = [('a', 'b'), ('a', 'd')]
        gr.remove_link_list(list2)

        self.assertFalse(gr.link_exists('a', 'b'), "Graph should NOT include a link between 'a' and 'b'")
        self.assertFalse(gr.link_exists('a', 'd'), "Graph should NOT include a link between 'a' and 'd'")

    def test_add_payload(self):
        gr = grl.Graph()

        gr.add_node_list([11, 12, 13, 14])
        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")

        list1 = [(11, 12), (12, 13), (11, 14), (13, 14)]
        gr.add_link_list(list1)

        self.assertFalse(gr.has_payload(11), "Graph should report that Node 11 DOES NOT have a payload")

        gr.add_payload(11, 'brand', 'toyota')
        gr.add_payload(11, 'miles', '50,000')

        gr.add_payload(12, 'brand', 'audi')
        gr.add_payload(12, 'miles', '30,300')

        gr.add_payload(13, 'brand', 'ford')
        gr.add_payload(13, 'miles', '12,000')

        self.assertTrue(gr.has_payload(11), "Graph should report that Node 11 has a non-zero payload")

        res1 = gr.get_payload(11)
        res2 = gr.get_payload(12)

        self.assertEquals(res1['brand'], 'toyota', "Node 11 should show a toyota car")
        self.assertEquals(res1['miles'], '50,000', "Node 11 should show a car with 50,000 miles")

        self.assertEquals(res2['brand'], 'audi', "Node 12 should show an audi car")
        self.assertEquals(res2['miles'], '30,300', "Node 12 should show a car with 30,300 miles")

        brand_value = gr.get_payload(13, name='brand')['brand']
        miles_value = gr.get_payload(13, name='miles')['miles']

        self.assertEquals(brand_value, 'ford')
        self.assertEquals(miles_value, '12,000')

    def test_remove_node_with_payload(self):
        gr = grl.Graph()

        gr.add_node_list([11, 12, 13, 14])
        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")

        list1 = [(11, 12), (12, 13), (11, 14), (13, 14)]
        gr.add_link_list(list1)

        gr.add_payload(11, 'brand', 'toyota')
        gr.add_payload(11, 'miles', '50,000')

        gr.add_payload(12, 'brand', 'audi')
        gr.add_payload(12, 'miles', '30,300')

        gr.remove_node(11)

        res1 = gr.get_payload(11)
        self.assertEquals(len(res1), 0, "If node does not exist function returns an empty dictionary")

    def test_is_payload_correct(self):
        payload1 = {'fname': 'sam', 'lname': 'davids', 'age': '24'}
        payload2 = {'fname': 'sam', 'lname': 'davids', 'age': 24}
        payload3 = {'fname': 'sam', 'lname': 'davids', 'age': '24', 'hobbies': ['biking', 'reading']}

        gr = grl.Graph()

        self.assertTrue(gr.is_payload_correct(payload1), "Function shoudl return True for payload1")
        self.assertFalse(gr.is_payload_correct(payload2), "Function shoudl return False for payload2")
        self.assertFalse(gr.is_payload_correct(payload3), "Function shoudl return False for payload3")

    def test_add_payload_multiple(self):
        payload1 = {'fname': 'sam', 'lname': 'davids', 'age': '24'}
        payload2 = {'fname': 'sam', 'lname': 'davids', 'age': 24}
        payload3 = {'fname': 'sam', 'lname': 'davids', 'age': '24', 'hobbies': ['biking', 'reading']}

        gr = grl.Graph()

        gr.add_node_list([11, 12, 13, 14])
        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")

        gr.add_payload_multiple(11, payload1)
        gr.add_payload_multiple(12, payload2)
        gr.add_payload_multiple(13, payload3)

    def test_payload_size(self):
        payload1 = {'fname': 'sam', 'lname': 'davids', 'age': '24'}

        gr = grl.Graph()

        gr.add_node_list([11, 12, 13, 14])
        self.assertEquals(gr.size(), 4, "Graph should report 4 nodes")

        gr.add_payload_multiple(11, payload1)
        number_pairs = gr.get_payload_size(11)

        self.assertEquals(number_pairs, 3, "Graph should report 3 name/value pairs for node 11")

    def test_dfs_traverse(self):
        gr = grl.Graph()

        nodes = range(7)

        gr.add_node_list(nodes)
        gr.add_link_list([(0, 3), (0, 2), (2, 4), (0, 1), (1, 5), (5, 4), (5, 6), (3, 4), (1, 6)])

        res = gr.dfs_traverse(0)

        self.assertEquals(len(res), 7, "DFS result should return all 7 nodes")

        set_nodes = set(nodes)
        set_res = set(res)

        set_diff = set_nodes.difference(set_res)

        self.assertEquals(len(set_diff), 0, "DFS result should contain the same node IDs as the original")

    def test_bfs_traverse(self):
        gr = grl.Graph()

        nodes = range(7)

        gr.add_node_list(nodes)
        gr.add_link_list([(0, 3), (0, 2), (2, 4), (0, 1), (1, 5), (5, 4), (5, 6), (3, 4), (1, 6)])

        res = gr.bfs_traverse(0)

        self.assertEquals(len(res), 7, "BFS result should return all 7 nodes")

        set_nodes = set(nodes)
        set_res = set(res)

        set_diff = set_nodes.difference(set_res)

        self.assertEquals(len(set_diff), 0, "BFS result should contain the same node IDs as the original")

    def test_is_connected_graph(self):
        gr = grl.Graph()

        nodes = range(7)

        gr.add_node_list(nodes)
        gr.add_link_list([(0, 3), (0, 2), (2, 4), (0, 1), (1, 5), (5, 4), (5, 6), (3, 4), (1, 6)])

        self.assertTrue(gr.is_connected_graph(0), "Function should determine that graph is a connected graph")

        gr.remove_node(6)

        self.assertEquals(gr.size(), 6, "Graph should have 6 nodes after removal of a node")

        gr.add_node(9)

        self.assertEquals(gr.size(), 7, "Graph should have 7 nodes after adding one node")

        self.assertFalse(gr.is_connected_graph(0), "Function should determine that graph is not a connected graph")

    def test_get_neighbors(self):
        gr = grl.Graph()

        nodes = range(7)

        gr.add_node_list(nodes)
        gr.add_link_list([(0, 3), (0, 2), (2, 4), (0, 1), (1, 5), (5, 4), (5, 6), (3, 4), (1, 6)])

        n_list = gr.get_neighbors(5)

        self.assertEquals(len(n_list), 3, "Function should report 3 neighbors for node 5")

        self.assertIn(1, n_list, "Function should report that node 1 is a neighbor of node 5")
        self.assertIn(4, n_list, "Function should report that node 4 is a neighbor of node 5")
        self.assertIn(6, n_list, "Function should report that node 6 is a neighbor of node 5")

    def test_get_depth_layer_nodes(self):
        ngr = grl.Graph()

        ngr.add_node_list(range(10))
        ngr.add_link_list([(0, 1), (0, 2), (0, 3), (1, 6), (1, 5), (2, 4), (5, 6)])
        ngr.add_link_list([(3, 4), (6, 7), (5, 8), (4, 8), (4, 9), (9, 8), (4, 5)])

        res0 = ngr.get_depth_layer(0, 0)

        self.assertEquals(len(res0['nodes']), 1, "Layer 0 should have only one node")
        self.assertEquals(res0['nodes'][0], 0, "The only node in layer 0 is the start node, which here it is 0")

        res1 = ngr.get_depth_layer(0, 1)
        self.assertEquals(len(res1['nodes']), 3, "Layer 1 should have 3 nodes")

        set1 = set([1, 2, 3])
        set2 = set(res1['nodes'])
        num_diff = len(set1.difference(set2))

        self.assertEquals(num_diff, 0, "Layer 1 should have nodes 1, 2, and 3")

        res2 = ngr.get_depth_layer(0, 2)
        self.assertEquals(len(res2['nodes']), 3, "Layer 2 should have 3 nodes")

        set3 = set([4, 5, 6])
        set4 = set(res2['nodes'])
        num_diff = len(set3.difference(set4))

        self.assertEquals(num_diff, 0, "Layer 2 should have nodes 4, 5, and 6")

        res10 = ngr.get_depth_layer(0, 10)
        self.assertEquals(res10['layer'], 3, "The maximum non-zero layer in this graph should be 3")

        set5 = set([7, 8, 9])
        set6 = set(res10['nodes'])
        num_diff = len(set5.difference(set6))

        self.assertEquals(num_diff, 0, "A request for layer 10 should return layer 3 nodes: 7, 8, 9")

    def test_is_node_in_layer(self):
        ngr = grl.Graph()

        ngr.add_node_list(range(10))
        ngr.add_link_list([(0, 1), (0, 2), (0, 3), (1, 6), (1, 5), (2, 4), (5, 6)])
        ngr.add_link_list([(3, 4), (6, 7), (5, 8), (4, 8), (4, 9), (9, 8), (4, 5)])

        self.assertTrue(ngr.is_node_in_layer(0, 0, 0), "Node 0 should be in layer 0")

        self.assertTrue(ngr.is_node_in_layer(0, 3, 1), "Node 3 should be in layer 1")

        self.assertTrue(ngr.is_node_in_layer(0, 5, 2), "Node 5 should be in layer 2")

        self.assertFalse(ngr.is_node_in_layer(0, 8, 2), "Node 8 should not be in layer 0")

    def test_shortest_path(self):
        tgr = grl.Graph()

        tgr.add_node_list(range(10))
        tgr.add_link_list([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 7), (4, 6), (5, 6), (6, 7)])
        tgr.add_link_list([(4, 8), (6, 8), (6, 9), (8, 9)])

        res0 = tgr.shortest_path(0, 0)
        self.assertEquals(len(res0), 1, "Shortest path for a single node should have a length of one")
        self.assertEquals(res0[0], 0, "Shortest path for a single node should have this node as the only element")

        res1 = tgr.shortest_path(0, 1)
        self.assertEquals(len(res1), 2, "Shortest path for node 1 includes 2 nodes")

        res2 = tgr.shortest_path(0, 2)
        self.assertEquals(len(res2), 2, "Shortest path for node 2 includes 2 nodes")

        res3 = tgr.shortest_path(0, 3)
        self.assertEquals(len(res3), 2, "Shortest path for node 3 includes 2 nodes")

        res4 = tgr.shortest_path(0, 4)
        self.assertEquals(len(res4), 3, "Shortest path for node 4 includes 3 nodes")

        res5 = tgr.shortest_path(0, 5)
        self.assertEquals(len(res5), 3, "Shortest path for node 5 includes 3 nodes")

        res6 = tgr.shortest_path(0, 6)
        self.assertEquals(len(res6), 4, "Shortest path for node 6 includes 4 nodes")

        res7 = tgr.shortest_path(0, 7)
        self.assertEquals(len(res7), 3, "Shortest path for node 7 includes 3 nodes")

        res8 = tgr.shortest_path(0, 8)
        self.assertEquals(len(res8), 4, "Shortest path for node 8 includes 4 nodes")

        res9 = tgr.shortest_path(0, 9)
        self.assertEquals(len(res9), 5, "Shortest path for node 9 includes 5 nodes")









