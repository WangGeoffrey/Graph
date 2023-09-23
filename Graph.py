from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple
from functools import reduce

class Node():
    
    def __init__(self, index: int) -> None:
        self.index = index
        self.neighbors: Set[Node] = set()
    
    def attach(self, observer: Node) -> None:
        self.neighbors.add(observer)
    
    def dettach(self, observer: Node) -> None:
        self.neighbors.remove(observer)

class Edge():
    
    def __init__(self, node1: Node, node2: Node, index: int, weight: int) -> None:
        self.index = index
        self.vertices: Tuple[Node, Node] = (node1, node2) # for directed edge: (leaving node, entering node)
        self.weight = weight
    
    @abstractmethod
    def remove(self) -> None:
        pass

class UEdge(Edge):
    
    # when edge is removed
    def remove(self) -> None:
        self.vertices[0].dettach(self.vertices[1])
        self.vertices[1].dettach(self.vertices[0])

class DEdge(Edge):
    
    # when edge is removed
    def remove(self) -> None:
        self.vertices[0].dettach(self.vertices[1])

class Graph(ABC):
    
    def __init__(self) -> None:
        self._matrix: List[List[int]] = []
        self._nodes: List[Node] = []
        self._edges: List[Edge] = []
    
    @property
    @abstractmethod
    def matrix(self) -> List[List[int]]:
        pass
    
    @matrix.setter
    @abstractmethod
    def matrix(self, matrix: List[List[int]]) -> None:
        pass
    
    @property
    def nodes(self) -> List[Node]:
        return self._nodes.copy()
    
    @property
    def edges(self) -> List[Edge]:
        return self._edges.copy()
    
    def add_node(self) -> None:
        self._nodes.append(Node(len(self._nodes)))
        self._matrix.append([0] * len(self._edges))
    
    def remove_node(self, node: Node) -> None:
        node_row = self._matrix.pop(self._nodes.index(node)) # remove node in matrix
        for edge_index in range(len(node_row)-1, -1, -1):
            if node_row[edge_index]: # true if edge connecting another node
                self.remove_edge(self._edges[edge_index])
        self._nodes.remove(node) # remove node from list of nodes
        for node in self._nodes[node.index:]:
            node.index -= 1
    
    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        pass
    
    def remove_edge(self, edge: Edge) -> None:
        edge_index = self._edges.index(edge)
        for node_row in self._matrix:
            node_row.pop(edge_index) # remove edge in matrix
        self._edges.remove(edge) # remove edge from list of edges
        edge.remove()
        for edge in self._edges[edge.index:]:
            edge.index -= 1
    
    def get_edge(self, node1: Node, node2: Node) -> Edge:
        # and matrix rows of nodes to get index of connecting edge
        edge_index = [a&b for a, b in zip(self._matrix[node1.index], self._matrix[node2.index])].index(1)
        return self._edges[edge_index]
    
    def valid_matrix(self, matrix: List[List[int]]) -> bool:
        if not matrix:
            return False
        if matrix == [[]]: # graph is a single node
            return True
        edges = len(matrix[0])
        for row in matrix: # check if matrix n by m
            if len(row) != edges:
                return False
        transposeMatrix = list(map(list, zip(*matrix)))
        for edge_row in transposeMatrix: # check if each edge is correct
            edge = [weight for weight in edge_row if weight]
            if len(edge) != 2: # has more/less than two nonzero values
                return False
            if type(self) == Undirected_Graph:
                if edge[0] != edge[1]: # two values are not the same
                    return False
            elif type(self) == Directed_Graph:
                if sum(edge): # sum of two values not zero
                    return False
            else:
                return False
        else:
            return True

    def clear(self) -> None:
        self.__init__()

# decorator to not run method if graph is empty
def empty_graph(func):
    def wrapper(graph: Graph):
        if graph.matrix:
            return func(graph)
        else:
            print(f'Graph is empty')
    return wrapper

class Undirected_Graph(Graph):
    
    @property
    def matrix(self) -> List[List[int]]:
        return self._matrix.copy()
    
    @matrix.setter
    def matrix(self, matrix: List[List[int]]) -> None:
        if not self.valid_matrix(matrix):
            return
        self.clear()
        for _ in matrix:
            self.add_node()
        for edge_index in range(len(matrix[0])):
            self.add_edge(*[self._nodes[node_index] for node_index in range(len(matrix)) if (value := matrix[node_index][edge_index]) if (weight := value)]+[weight])
    
    def add_edge(self, node1: Node, node2: Node, weight=1) -> None:
        self._edges.append(UEdge(node1, node2, len(self._edges), weight))
        node1.attach(node2)
        node2.attach(node1)
        for node_index in range(len(self._nodes)):
            self._matrix[node_index].append(int(self._nodes[node_index] in [node1, node2])) # 1 if node a vertice of edge, zero otherwise
    
    # returns set of connected nodes using depth first search
    def connected_graph(self, current: Node, visited: Set[Node]):
        for node in current.neighbors:
            if node not in visited:
                visited.add(node)
                visited = self.connected_graph(node, visited)
        return visited
    
    # check if graph is connected - every pair of nodes connected by some path
    def is_connected_graph(self) -> bool:
        visited = self.connected_graph(self.nodes[0], {self.nodes[0]})
        return visited == set(self._nodes)

    # find the minimum spanning tree of the graph using Kruskal's algorithm
    @empty_graph
    def mst(self) -> Set[UEdge]:
        if not self.is_connected_graph():
            return None
        mst = set()
        forest = []
        nodes = set(self._nodes)
        edges = self._edges.copy()
        edges.sort(key=lambda x: x.weight)
        while nodes: # while all nodes not in mst
            edge = edges.pop(0)
            node_pair = set(edge.vertices)
            index = 0
            if node_pair.issubset(nodes): # both nodes not in mst
                mst.add(edge)
                nodes = nodes.difference(node_pair)
                forest.append(node_pair)
            else:
                for tree in forest:
                    if node_pair.issubset(tree): # both nodes in mst
                        break
                    elif node_pair.intersection(tree): # one node in mst
                        if index: # both nodes in different mst
                            forest[index] = tree.union(forest.pop(forest.index(tree)))
                            break
                        mst.add(edge)
                        index = forest.index(tree)
                        forest[index] = tree.union(node_pair)
                else: # only one node was in mst
                    nodes = nodes.difference(node_pair)
        return mst
    
    # find a hamiltonian cycle in the graph using backtracking algorithm
    @empty_graph
    def hamiltonian_cycle(self) -> Set[Edge]:
        start = self._nodes[0]
        if cycle := self.h_cycle(start, start, set(self.nodes).difference({start}), set()): # (start node, current node in recursion, set of nodes not visited, set of edges in cycle)
            return cycle
    
    # recursive method for hamiltonian cycle
    def h_cycle(self, start: Node, current: Node, not_visited: Set[Node], cycle: Set[Edge]):
        if not not_visited and start in current.neighbors: # if cycle found
            return cycle.union({self.get_edge(current, start)})
        else:
            for node in current.neighbors:
                if node in not_visited:
                    if check := self.h_cycle(start, node, not_visited.difference({node}), cycle.union({self.get_edge(current, node)})):
                        return check
    
    # find a max matching of the graph using augmenting paths and Berge's Lemma
    @empty_graph
    def max_matching(self) -> Set[Edge]:
        matching: Set[Edge] = set() # edges in matching
        exposed = set(self._nodes) # nodes not in matching
        while exposed: # while there are unmatched nodes
            # for node in exposed:
            #     # if path := self.augmenting_pathB(node, matching, exposed, [], {node: True}): # if augmenting path found
            #     if path := self.augment_blossom(node.index, list(range(len(self._nodes))), matching, {n.index for n in exposed}, self.matrix):
            #         break
            # else:
            #     break # no more augmenting paths
            # alternating_path = set(path).difference(matching) # remove edges in matching from path to get alternating path
            # matching = matching.difference(set(path)).union(alternating_path) # xor edges in matching with edges in alternating path
            # exposed.remove(node)
            # exposed = exposed.difference(set(path[-1].vertices))
            
            if not (path := self.aug_path(matching, exposed)): # if no more augmenting paths
                break
            exposed = exposed.difference(set(path[0].vertices))
            if len(path) == 1:
                matching.add(path[0])
            else:
                exposed = exposed.difference(set(path[-1].vertices))
                alternating_path = set(path).difference(matching) # remove edges in matching from path to get alternating path
                matching = matching.difference(set(path)).union(alternating_path) # xor edges in matching with edges in alternating path
        return matching
    
    # recursive method for finding augmenting path
    # label = True if start node or edge connecting node to path is in matching
    # label = False if edge connecting node to path is not in matching
    def augmenting_path(self, current: Node, matching: Set[Edge], exposed: Set[Node], path: Set[Edge], label: bool) -> List[Edge]:
        for node in current.neighbors:
            edge = self.get_edge(current, node)
            if label and node in exposed: # if augmenting path found
                return path + [edge]
            if label or edge in matching: # if alternating edge
                if edge not in path: # if adding edge does not cause a cycle
                    if result := self.augmenting_path(node, matching, exposed, path + [edge], not label): # if augmenting path found
                        return result
    
    # recursive method for finding augmenting path using blossom algorithm
    # blossom - an odd-length cycle
    def augmenting_pathB(self, current: Node, matching: Set[Edge], exposed: Set[Node], path: List[Edge], label: Dict[Node, bool], blossom=None) -> List[Edge]:
        current_node = current # current later assigned to node in blossom if contacted blossom
        for node in current_node.neighbors:
            if blossom: # if graph contains a contracted blossom
                if current_node not in self._nodes: # if current node is a contracted blossom
                    # get node in contracted blossom connected to current node
                    current = blossom.intersection(node.neighbors).pop()
            edge = self.get_edge(current, node)
            if node in label: # if node covered by path
                if label[current] & label[node]: # if blossom encountered
                    new_blossom = {current} # set of nodes in blossom
                    contracted_path = path.copy() # path in contrated graph
                    # remove edges in blossom from contracted graph path
                    for path_edge in path[::-2]:
                        if node in set(path_edge.vertices): # if blossom removed
                            break
                        contracted_path.pop(-1)
                        new_blossom = new_blossom.union(set(contracted_path.pop(-1).vertices))
                    if blossom:
                        new_blossom = new_blossom.union(blossom)
                    contracted_node = Node(-1) # node of contracted blossom
                    neighbors = set()
                    for vertex in new_blossom:
                        neighbors = neighbors.union(vertex.neighbors)
                    neighbors = neighbors.difference(new_blossom)
                    for neighbor in neighbors:
                        contracted_node.attach(neighbor)
                    new_label = label.copy()
                    for blossom_node in new_blossom:
                        new_label[blossom_node] = True
                    # start new recursion with contracted graph
                    if result := self.augmenting_pathB(contracted_node, matching, exposed, contracted_path, new_label, new_blossom):
                        path.append(edge) # complete blossom cycle for finding connecting path
                        index = len(contracted_path) # index of edge before contraction
                        new_edge = result[index] # edge added after contraction
                        # add edges from blossom to contracted path to complete uncontracted path
                        for count, blossom_edge in enumerate(path[index:]):
                            # check direction path takes to connect outer vertice of matched edge in blossom to edge after contraction
                            if set(new_edge.vertices).intersection(blossom_edge.vertices):
                                if blossom_edge in matching:
                                    connecting_path = path[index:index + count+1]
                                else:
                                    connecting_path = path[:index + count:-1]
                                break
                        return result[:index] + connecting_path + result[index:]
                continue # adding edge will introduce cycle, check next edge
            if label[current] and node in exposed: # if augmenting path found
                return path + [edge]
            if label[current] or edge in matching: # if alternating edge
                label.update({node: not label[current]})
                if result := self.augmenting_pathB(node, matching, exposed, path + [edge], label, blossom): # if augmenting path found
                    return result
                label.pop(node)

    # current node, matching, exposed, path, label, nodes in current instance, bit matrix representing current instance
    def augment_blossom(self, current: int, nodes: List[int], matching: Set[Edge], exposed: Set[int], matrix: List[List[int]]) -> List[Edge]:
        start = current
        label = {current: True}
        path: List[Edge] = []
        key = None
        while True:
            if current != key: # if next edge
                # & matrix rows of other nodes with matrix row of current node to get nodes which share an edge (neighbors of current node)
                skip = nodes.index(current) # exclude checking with itself
                neighbors = (n for n in nodes[:skip] + nodes[skip+1:] if 1 in [a&b for a, b in zip(matrix[current], matrix[n])]) # generator of valid neighbors of current node
                key = current
            if (node := next(neighbors, None)) == None: # if no other neighbors
                return None
            edge = self._edges[[a&b for a, b in zip(matrix[current], matrix[node])].index(1)]
            if node in label: # if visited node
                if not (label[current] and label[node]): # if not a blossom (label of nodes are not both True)
                    continue # edge in path, check another edge
                # get nodes and edges in blossom
                blossom = {node} # set of nodes in blossom
                for path_edge in path[::-2]:
                    blossom = blossom.union({n.index for n in path_edge.vertices})
                    if matrix[node][path_edge.index]: # if node is a vertex of edge (entire blossom found)
                        blossom_edges = path[path.index(path_edge):] # list of edges in blossom
                        break
                # xor matrix row of nodes in blossom to get matrix row of contracted node
                contracted = [0] * len(self._edges)
                for node_index in blossom:
                    contracted = [a^b for a, b in zip(contracted, matrix[node_index])]
                matrix.insert(len(self._nodes), contracted)
                if start in blossom: # if start node in blossom
                    start = len(self._nodes) # make recursive call with blossom node
                if not (result := self.augment_blossom(start, [n for n in nodes if n not in blossom] + [len(self._nodes)], matching, exposed, matrix)): # if augmenting path not found
                    return None
                #     c-d=e     Edges part of matching a=b, d=e and g=h
                # 0-a=b<  |     When uncontracting blossom, find if path goes cw or ccw
                #     f-g=h     e.g. c exposed - then 0-a-b-g-h-e-d-c (ccw) | f exposed - then 0-a-b-d-e-h-g-f (cw)
                
                # check if gap in path caused by uncontracted blossom
                for index in range(len(result)-1):
                    if not set(result[index].vertices).intersection(set(result[index+1].vertices)): # if edges are not connected
                        index += 1
                        node1, node2 = result[index].vertices
                        temp = node1 if node2 in result[index+1].vertices else node2
                        # complete path with edges in blossom
                        for path_edge in result:
                            if temp in path_edge.vertices:
                                if label[temp.index]:
                                    connecting_path = blossom_edges[:blossom_edges.index(path_edge)]
                                else:
                                    connecting_path = blossom_edges[blossom_edges.index(path_edge):]
                                return result[index-1:] + connecting_path + result[:index-1]
                else:
                    return result # no gap in path caused by blossom
            elif label[current] and node in exposed: # if augmenting path found
                return path + [edge] # return augmenting path
            elif label[current] or edge in matching: # if alternating edge
                # update path with edge
                label.update({node: not label[current]})
                path.append(edge)
                current = node

    def aug_path(self, matching: Set[Edge], exposed: Set[Node]) -> List[Edge]:
        marked: Dict[Node, Tuple[Node, Node]] = {} # key = node, value = tuple(previous node, root node)
        stack: List[Node] = []
        for node in exposed:
            marked.update({node: (None, node)})
            stack.append(node)
            while stack: # while there are branches of tree with exposed node as root
                node = stack.pop(-1)
                for neighbor in node.neighbors:
                    if neighbor in exposed and marked[node] != neighbor: # if path found
                        # construct path to return
                        path = []
                        while node is not None:
                            path.append(self.get_edge(node, neighbor))
                            neighbor, node = node, marked[node][0]
                        return path
                    if neighbor not in marked: # if neighbor not in tree
                        # add pair of alternating edges
                        for edge in matching:
                            if neighbor in edge.vertices: # if neighbor a vertice of edge in matching
                                next = edge.vertices[(edge.vertices.index(neighbor)+1)%2] # get next node following edge in matching
                                break
                        marked.update({neighbor: (node, marked[node][1])})
                        marked.update({next: (neighbor, marked[neighbor][1])})
                        stack.append(next)

class Directed_Graph(Graph):
    
    @property
    def matrix(self) -> List[List[int]]:
        return self._matrix.copy()
    
    @matrix.setter
    def matrix(self, matrix: List[List[int]]) -> None:
        if not self.valid_matrix(matrix):
            return
        self.clear()
        for _ in matrix:
            self.add_node()
        for edge_index in range(len(matrix[0])):
            for node_index in range(len(matrix)):
                if value := matrix[node_index][edge_index]:
                    if value > 0:
                        weight = value
                        node2 = self._nodes[node_index]
                    else:
                        node1 = self._nodes[node_index]
            self.add_edge(node1, node2, weight)
    
    def add_edge(self, node1: Node, node2: Node, weight=1) -> None:
        self._edges.append(DEdge(node1, node2, len(self._edges), weight))
        node1.attach(node2)
        for node_index, node in enumerate(self._nodes):
            if node in [node1, node2]:
                if node == node2:
                    self._matrix[node_index].append(1) # entering node
                else:
                    self._matrix[node_index].append(-1) # leaving node
            else:
                self._matrix[node_index].append(0)
