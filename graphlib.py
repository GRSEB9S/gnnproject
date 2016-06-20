""" Library for building and using undirected graphs. It defines a generic Graph class and its methods

    A graph G(N, L) is a collection of nodes (also called vertices) that connect to each other using
    links (also called edges). A graph is called connected if there are no isolated nodes. A graph
    is called undirected if all links are bi-directional. A graph is called simple if it does not have
    links from a node to the same node, and if there can be at most one link from node to node.

    This library supports simple undirected graphs (connected or non-connected).

    In this implementation a graph is defined as a set of dictionary objects indexed by node
    identifiers (IDs). One dictionary object defines the links (edges) associated with a node. A link is
    defined by its two terminating nodes.

    Another dictionary object defines a payload that can be associated with graph nodes. The payload
    is a collection of name/value pairs. Users define the name/value pairs that should be inserted
    in the payload.

"""

import os
import sys

__author__ = "Edwin Heredia"
__copyright__ = "Copyright 2016"
__license__ = "Not ready for distribution"
__version__ = "0.0.8"
__status__ = "In development"


class Graph(object):
    def __init__(self):
        # Define private variables
        self.__data = {}     # dictionary object that maps a node with the set of neighbor nodes
        self.__payload = {}  # dictionary object that maps a node with a payload

    @staticmethod
    def is_payload_correct(data):
        """
        Checks if the data argument contains a dictionary with name/value pairs, and if the names
        and values are strings.
        Args:
            data: A dictionary object
        Returns:
            True if the data dictionary carries name/value pairs only, and they are strings. Otherwise,
            it returns False.
        """
        keys = data.keys()
        values = data.values()

        bool_list1 = [isinstance(val, basestring) for val in keys]
        bool_list2 = [isinstance(val, basestring) for val in values]

        if all(bool_list1) and all(bool_list2):
            return True
        else:
            return False

    def size(self):
        """
        Provides the number of nodes in the graph
        Args:
        Return:
            The number of nodes in the graph
        """
        return len(self.__data)

    def node_exists(self, node_id):
        """
        Verifies if a node identified by an ID already exists in the graph
        Args:
            node_id: (integer or string) A unique node identifier
        Returns:
            True if a node with such identifier exists. False otherwise.
        """
        if node_id in self.__data.keys():
            return True
        else:
            return False

    def has_payload(self, node_id):
        """
        Vefifies if a node has a non-zero payload
        Args:
            node_id: (integer or string) A unique node identifier
        Returns:
            True if the referenced node has a non-zero payload. Otherwise it returns False.
        """
        if node_id not in self.__data.keys():
            return False
        else:
            if node_id not in self.__payload.keys():
                return False
            else:
                return len(self.__payload[node_id]) != 0

    def link_exists(self, init_id, dest_id):
        """
        Verifies if a certain link between two nodes already exists in the graph
        Args:
            init_id: (integer or string) The ID that identifies the first node
            dest_id: (integer or string) The ID that identifies the second node
        Returns
            True if the link exists. False otherwise
        """
        if init_id in self.__data.keys() and dest_id in self.__data.keys():
            links1 = self.__data[init_id]
            links2 = self.__data[dest_id]

            if dest_id in links1 and init_id in links2:
                return True
            else:
                return False

    def get_nodes(self):
        """
        Retrieves a list of nodes in the graph
        Returns:
            A list with the node IDs
        """
        return self.__data.keys()

    def get_links(self, node_id):
        """
        Retrieves the list of neighbor nodes for a node identified by its ID
        Args:
            node_id: (integer or string) A unique node identifier
        Returns:
            A list with the IDs of the neighboring nodes or None if the target node does not exist in the graph.
        """
        if node_id in self.__data.keys():
            return self.__data[node_id]

    def add_node(self, node_id):
        """
        Adds a node with no links to the graph
        Args:
            node_id: (integer or string) A unique node identifier
        """
        if node_id not in self.__data.keys():
            self.__data[node_id] = []

    def add_node_list(self, node_list):
        """
        Adds a group of nodes represented as elements in node_list
        Args:
            node_list: A list of node ID values
        """
        for nd in node_list:
            if nd not in self.__data.keys():
                self.__data[nd] = []

    def remove_node_list(self, node_list):
        """
        Removes nodes from a node list
        Args:
            node_list: A list of node ID values
        """
        for nd in node_list:
            if nd in self.__data.keys():
                self.remove_node(nd)

    def add_link_list(self, link_list):
        """
        Adds a list of links to the graph
        Args:
            link_list: A list of tuples where each tuple defines a link between two nodes
        """
        for edge in link_list:
            self.add_link(edge[0], edge[1])

    def remove_link_list(self, link_list):
        """
        Removes a list of links from the graph
        Args:
            link_list: A list of tuples where each tuple defines a link between two nodes
        """
        for edge in link_list:
            self.remove_link(edge[0], edge[1])

    def remove_node(self, node_id):
        """
        Removes an existing node from the graph
        Args:
            node_id: (integer or string) The ID of the node to be removed
        """
        if node_id in self.__data.keys():
            # Retrieve all the neighbor nodes
            nbors = list(self.__data[node_id])   # need to make an actual duplicate to avoid referencial changes

            # Remove links between the selected node and its neighbors
            for nb in nbors:
                self.remove_link(node_id, nb)

            # Remove the entry for the selected node
            del self.__data[node_id]

        if node_id in self.__payload.keys():
            del self.__payload[node_id]

    def remove_link(self, init_id, dest_id):
        """
        Removes a link between two nodes identified by their IDs
        Args:
            init_id: (integer or string) The ID of the first node
            dest_id: (integer or string) The ID of the second node
        """
        if init_id in self.__data.keys() and dest_id in self.__data.keys():
            if dest_id in self.__data[init_id]:
                self.__data[init_id].remove(dest_id)

            if init_id in self.__data[dest_id]:
                self.__data[dest_id].remove(init_id)

    def add_link(self, init_id, dest_id):
        """
        Adds a link between two nodes identified by their IDs
        Args:
            init_id: (integer or string) The ID of the first node
            dest_id: (integer or string) The ID of the second node
        """
        if init_id in self.__data.keys() and dest_id in self.__data.keys():
            self.__data[init_id].append(dest_id)
            self.__data[dest_id].append(init_id)

    def __str__(self):
        """
        Defines a string displaying graph components for use in print statements
        Returns:
            A string that when printed it shows all the data available in a graph
        """
        txt = "--------------------------------------\n"
        txt += "Node     List of neighbor nodes\n"
        txt += "--------------------------------------\n"
        for key in self.__data.keys():
            txt += str(key) + "        " + str(self.__data[key]) + "\n"

        txt += "......................................\n"
        txt += "Node\tName\tValue\n"
        txt += "......................................\n"

        for node in self.__payload.keys():
            for name in self.__payload[node].keys():
                value = self.__payload[node][name]
                txt += str(node) + "\t" + name + "\t" + value + "\n"

        return txt

    def dfs_traverse(self, init_id):
        """Use the Depth-First Search (DFS) algorithm for graph traversal
        Args:
            init_id:  (integer or string) An identifier for the starting node
        Returns:
            A list of visited nodes in the order in which they have been visited
        """

        visited = [init_id]
        path = [init_id]
        self.__rec_dfs_traverse(init_id, visited, path)
        # print "visited nodes: ", visited
        return visited

    def __rec_dfs_traverse(self, current, visited, path):
        """(Private method) Process graph node and continue to next node recursively
        using the DFS strategy.

        Args:
            current: (integer or string) The current node ID which is being processed
            visited: (list) A list of visited nodes identified by their IDs
            path: (list) A list of nodes representing the traversal path
        """

        # define when to terminate recursion
        if len(path) == 0:
            return

        # determine the links (edges) for the current node
        links = self.__data[current]

        # determine if any of the links has not been visited
        found = False
        selected = None
        for link in links:
            if link not in visited:
                found = True
                selected = link
                break

        # If a link has not been visited then vist the node and repeat process (recursion)
        if found:
            visited.append(selected)
            path.append(selected)
            self.__rec_dfs_traverse(selected, visited, path)

        # If all links have been visited then backtrack to previous entry in path and repeat process (recursion)
        else:
            path.pop()
            if len(path) != 0:
                new_current = path[len(path) - 1]
            else:
                new_current = None  # If the path is empty there is no current node

            self.__rec_dfs_traverse(new_current, visited, path)

    def bfs_traverse(self, init_id):
        """Use the Breadth-First Search (BFS) algorithm for graph traversal
        Args:
            init_id:  (integer or string) An identifier for the starting node
        Returns:
            A list of visited nodes in the order in which they have been visited
        """

        visited = [init_id]
        stack = [init_id]
        self.__rec_bfs_traverse(visited, stack)
        # print "visited nodes: ", visited
        return visited

    def __rec_bfs_traverse(self, visited, stack):
        """(Private method) Process graph node and continue to next node recursively
        using the BFS strategy.

        Args:
            visited: (list) A list of visited nodes identified by their IDs
            stack: (list) A list of nodes representing the processed nodes
        """

        # define when to terminate recursion
        if len(stack) == 0:
            return

        # get current working node from a LIFO stack and find its neighbors
        current = stack.pop()
        links = self.__data[current]

        # find the subset of neighbors that have not yet been visited
        unvisited = []
        for link in links:
            if link not in visited:
                unvisited.append(link)

        # add univisted neighbors to the visited list and to the stack
        if len(unvisited) != 0:
            for unv in unvisited:
                visited.append(unv)
                stack.append(unv)
            self.__rec_bfs_traverse(visited, stack)
        # if there are no unvisited neighbors then continue recursion
        else:
            self.__rec_bfs_traverse(visited, stack)

    def save(self, filepath):
        """
        Saves the graph data to a file identified by its path. The file is a text file.
        File format: Each line in the file lists a node and its neighbors with tab separations

        TBD: Future work needs to address parsing errors
        Args:
            filepath: An absolute or relative path
        """
        with open(filepath, "w") as fp:
            # Save topology information into the file
            for node, nbors in self.__data.items():
                nbors_string_list = [str(val) for val in nbors]
                nbors_text = "\t".join(nbors_string_list)
                single_line = "T" + "\t" + str(node) + "\t" + nbors_text + "\n"
                fp.write(single_line)

            # Save payload information into the file
            for node in self.__payload.keys():
                for name in self.__payload[node].keys():
                    value = self.__payload[node][name]
                    one_line = "P" + "\t" + str(node) + "\t" + str(name) + "\t" + str(value) + "\n"
                    fp.write(one_line)

    def read(self, filepath):
        """
        Reads graph data from a file identified by its path. The file is a text file.
        File format: Each line in the file lists a node and its neighbors with tab separations

        TBD: Future work needs to address parsing errors
        Args:
            filepath: An absolute or relative path
        Returns:
            True if the graph has been uploaded and false if there is an error. If the calling graph
            is not empty this method returns False (graph data is not replaced). If the file does
            not exist this method returns False.
        """

        if self.size() != 0:
            return False

        if not os.path.isfile(filepath):
            return False

        with open(filepath, "r") as fp:
            lines = fp.readlines()

            for line in lines:
                # remove newline character and split using tabs
                line_list = line.strip("\n").split("\t")

                if line_list[0] == 'T':
                    # process a line that carries topology information
                    topology_info = line_list[1:]
                    self.__upload_topology(topology_info)

                elif line_list[0] == 'P':
                    # process a line that carries payload information
                    payload_info = line_list[1:]
                    self.__upload_payload(payload_info)

                else:
                    print "[Error] Program has reached an unused code path"
                    sys.exit(1)
        return True

    def __upload_topology(self, info_list):
        """
        (Private method) Uploads a line of topology information to the graph. The line
        has been pre-processed and exists as a list of string tokens.
        Args:
            info_list: A list of strings representing a line of topology data extracted from a file
        """
        ret_node = info_list[0]
        ret_nbors = info_list[1:]

        # discover if the ID format is integers or strings
        id_format = 'integer' if ret_node.isdigit() else 'string'

        if id_format == 'integer':
            node = int(ret_node)
            nbors = [int(val) for val in ret_nbors]

        else:
            node = ret_node
            nbors = ret_nbors

        # add retrieved node to graph
        self.add_node(node)

        # create links from retrieved neighbor nodes
        for nbor in nbors:
            self.add_link(node, nbor)

    def __upload_payload(self, info_list):
        """
        (Private method) Uploads a line of payload information to the graph. The line
        has been pre-processed and exists as a list of string tokens.
        Args:
            info_list: A list of 3 strings representing node ID, name, and value
        """
        ret_node = info_list[0]
        name = info_list[1]
        value = info_list[2]

        # discover if the ID format is integers or strings
        id_format = 'integer' if ret_node.isdigit() else 'string'

        node = int(ret_node) if id_format == "integer" else ret_node

        self.add_payload(node, name, value)

    def add_payload(self, node_id, name, value):
        """
        Add payload information to any node in the graph
        Args:
            node_id: (string or number) The ID of the target node
            name: (string) The name of the attribute added to the node
            value: The value of the attribute added to the node
        """
        if node_id in self.__data.keys():
            # check if there is an entry for node_id in the payload dictionary. If not create one.
            if node_id not in self.__payload.keys():
                self.__payload[node_id] = {}

            # store the entered name/value pair in the node_id element in payload
            self.__payload[node_id][name] = value

    def add_payload_multiple(self, node_id, data):
        """
        Adds payload information to any node in the graph

        TBD: In the future we need to verify the syntax of data before adding it to
        the payload data structure

        Args:
            node_id: (integer or string) Identifies the target node for adding payload data
            data: (dictionary) A dictionary object with name/value pairs. Any name or value must be a string.
        """
        if node_id in self.__data.keys():
            if node_id not in self.__payload.keys():
                if self.is_payload_correct(data):
                    self.__payload[node_id] = data

    def get_payload(self, node_id, name=None):
        """
        Retrieve the payload assigned to a particular node
        Args:
            node_id: (string or number) The ID of the target node
            name: (string) If the call omits the name, the function returns all the payload. If the call
                           includes a name, the function returns an object with a single name/value pair.
        Return:
             If the node_id does not exist in the graph the function returns an empty dictionary object
             If the node_id exits but it does not have a payload, the function returns an empty dictionary object
             If the node_id exists and has a payload but the given name is not part of the payload, then the
                function returns an empty dictionary object
             If the node_id exits and has a payload and the call omits name, the function returns all the payload
                as a dictionary object
             If the node_id exists and has a payload and the call includes an available name, the function returns
             the requested name/value pair as a single element in a dictionary object
        """
        if node_id not in self.__data.keys():
            return {}
        if node_id not in self.__payload.keys():
            return {}

        if name is not None:
            if name not in self.__payload[node_id].keys():
                return {}
            else:
                return {name: self.__payload[node_id][name]}
        else:
            return self.__payload[node_id]

    def get_payload_size(self, node_id):
        """
        Retrieve the number of name/value pairs assigned as payload to a node identified
        by its node_id
        Args:
            node_id: (integer or string) The ID that identifies the target node
        Returns:
            The number of name/value pairs assigned as payload. The number can be zero. If
            the node_id does not exist in the graph, then the function returns -1

        """
        if node_id in self.__data.keys():
            if node_id in self.__payload.keys():
                return len(self.__payload[node_id])
            else:
                return 0
        else:
            return -1

    def is_connected_graph(self, node_id):
        """
        Determines if the graph sample is a fully connected graph, i.e. it has zero
        isolated nodes
        Args:
            node_id: (integer or string) A starting node from where to start the test
        Returns:
            True if graph is fully connected or False otherwise
        """

        visited = self.dfs_traverse(node_id)
        if len(visited) == self.size():
            return True
        else:
            return False

    def get_neighbors(self, node_id):
        """
        Returns a list of the neighbor nodes for the node specified by a node_id
        Args:
            node_id: (integer or string) The target node for this operation
        Returns:
            A list of node IDs or None if the node does not exist in the graph
        """
        if node_id in self.__data.keys():
            return self.__data[node_id]
        else:
            return None

    def get_depth_layer(self, node_id, layer):
        """
        Return the list of nodes at a certain depth level from the start node
        Args:
            node_id: (integer or string) The ID for the start node
            layer: (integer) The separation layer (number of hops) from the start node. The start node is
                at layer 0, the next nodes (one hop) are at layer 1, and so forth.
        Returns:
             A dictionary object with a layer and nodes fields. The 'layer' field indicates the maximum
             layer reached (which can be the same as the requested layer). The 'nodes' field carries the
             list of nodes at such layer.
        """
        if layer == 0:
            return {'layer': 0, 'nodes': [node_id]}

        finished = False
        layer_nodes = [node_id]
        visited = [node_id]
        layer_count = 0

        while not finished:
            layer_count += 1
            new_layer_nodes = []

            for nd in layer_nodes:
                for nid in self.get_neighbors(nd):
                    if nid not in visited:
                        visited.append(nid)
                        new_layer_nodes.append(nid)

            if len(new_layer_nodes) == 0:
                finished = True
                layer_count -= 1

            elif layer_count == layer:
                finished = True
                layer_nodes = new_layer_nodes

            else:
                layer_nodes = new_layer_nodes

        response = {'layer': layer_count,
                    'nodes': layer_nodes}

        return response

    def is_node_in_layer(self, start_node, target_node, layer):
        """
        Checks if a destination node exists at a given depth layer from the initiating node
        Args:
            start_node: The node ID for the initiating node. This node is at layer 0.
            target_node: The node ID for the target node. The function determines if this target node is at
                a given layer
            layer: An integer value representing the investigated layer
        Returns:
            True if the target node is at the specified layer. False otherwise. If the nodes are not
            graph members the function returns False.
        """
        if start_node not in self.__data.keys():
            return False

        if target_node not in self.__data.keys():
            return False

        result = self.get_depth_layer(start_node, layer)

        if result["layer"] < layer:
            return False
        else:
            if target_node in result["nodes"]:
                return True

    def shortest_path(self, init_id, dest_id):
        """Use a Breadth-First Search (BFS) strategy to obtain the shortest path
        between an initial node and a destination node

        Args:
            init_id:  An identifier for the starting node
            dest_id:  An identifier for the destination node
        Returns:
            A list of nodes defining the shortest path between the two nodes. An empty
            list indicates an error with the node identifiers.
        """
        if not (self.node_exists(init_id) and self.node_exists(dest_id)):
            return []

        if init_id == dest_id:
            return [init_id]

        # Initialize map of layers
        layer_count = 0
        vis_map = {init_id: layer_count}

        finished = False
        layer_nodes = [init_id]

        # Iterate through the graph to tag each node with a layer value
        while not finished:
            layer_count += 1
            new_layer_nodes = []

            for nd in layer_nodes:
                for nid in self.get_neighbors(nd):
                    if nid not in vis_map.keys():
                        vis_map[nid] = layer_count
                        new_layer_nodes.append(nid)

            if len(new_layer_nodes) == 0:
                finished = True
                layer_count -= 1

            else:
                layer_nodes = new_layer_nodes

        # Backtrack from destination to initial node to find path
        current = dest_id
        node_path = [dest_id]
        counter = 0
        while current != init_id and counter < 1000:
            nbors = self.get_neighbors(current)

            min_layer = self.size() + 1  # large number that exceeds max layer value
            best_parent = None
            for nd in nbors:
                if vis_map[nd] < min_layer:
                    min_layer = vis_map[nd]
                    best_parent = nd

            node_path.append(best_parent)
            current = best_parent
            counter += 1

        # Return reversed path
        return node_path[::-1]


if __name__ == "__main__":
    gr = Graph()

    for k in range(7):
        gr.add_node(k)

    gr.add_link(0, 3)
    gr.add_link(0, 2)
    gr.add_link(2, 4)
    gr.add_link(0, 1)
    gr.add_link(1, 5)
    gr.add_link(5, 4)
    gr.add_link(5, 6)
    gr.add_link(3, 4)
    gr.add_link(1, 6)

    print gr

    print "Traverse graph starting from node 0 using DFS: "
    visited1 = gr.dfs_traverse(0)
    print "DFS visited nodes: ", visited1

    print "Traverse graph starting from node 0 using BFS: "
    visited2 = gr.bfs_traverse(0)
    print "BFS visited nodes: ", visited2

    print "**********************************"

    gr.add_payload(0, 'fname', 'rob')
    gr.add_payload(0, 'lname', 'williams')

    gr.add_payload(1, 'fname', 'sam')
    gr.add_payload(1, 'lname', 'parker')

    gr.add_payload(2, 'fname', 'taylor')
    gr.add_payload(2, 'lname', 'swift')

    res0 = gr.get_payload(0)
    res1 = gr.get_payload(1)

    print "res0: " + str(res0)
    print "res1: " + str(res1)

    res2 = gr.get_payload(2, name="fname")
    print "res2: " + str(res2)

    gr.save("graph1.grf")

    grnew = Graph()
    grnew.read("graph1.grf")

    print grnew

    print "#####################################"

    tgr = Graph()

    tgr.add_node_list(range(10))
    tgr.add_link_list([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 7), (4, 6), (5, 6), (6, 7)])
    tgr.add_link_list([(4, 8), (6, 8), (6, 9), (8, 9)])

    print tgr

    s_path = tgr.shortest_path(0, 9)

    print "shortest path: ", s_path




