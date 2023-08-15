from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Set, Tuple

# = = = = Observer Design Pattern = = = = 

class Fixed_Subject(ABC):
    
    @abstractmethod
    def notify(self) -> None:
        pass

class Subject(Fixed_Subject):
    
    @abstractmethod
    def attach(self, observer: Observer) -> None:
        pass
    
    @abstractmethod
    def dettach(self, observer: Observer) -> None:
        pass
    
class Observer(ABC):
    
    @abstractmethod
    def update(self, subject: Subject) -> None:
        pass

# = = = = Observer Design Pattern = = = = 

class Node(Subject, Observer):
    
    def __init__(self) -> None:
        self.neighbors: Set[Node] = set()
    
    def attach(self, observer: Node) -> None:
        self.neighbors.add(observer)
    
    def dettach(self, observer: Node) -> None:
        self.neighbors.remove(observer)
    
    # when node is removed
    def notify(self) -> None:
        for node in self.neighbors:
            node.update(self)
    
    # update from removed node/edge
    def update(self, subject: Node) -> None:
        self.dettach(subject)

class Edge(Fixed_Subject):
    
    def __init__(self, node1: Node, node2: Node, weight) -> None:
        self.vertices: Tuple[Node, Node] = (node1, node2) # for directed edge: (leaving node, entering node)
        self.weight = weight

class UEdge(Edge):
    
    # when edge is removed
    def notify(self) -> None:
        self.vertices[0].update(self.vertices[1])
        self.vertices[1].update(self.vertices[0])

class DEdge(Edge):
    
    # when edge is removed
    def notify(self) -> None:
        self.vertices[0].update(self.vertices[1])

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
        self._nodes.append(Node())
        self._matrix.append([0] * len(self._edges))
    
    def remove_node(self, node: Node) -> None:
        node_row = self._matrix.pop(self._nodes.index(node)) # remove node in matrix
        for edge_index in range(len(node_row)-1, -1, -1):
            if node_row[edge_index]: # true if edge connecting another node
                self.remove_edge(self._edges[edge_index])
        self._nodes.remove(node) # remove node from list of nodes
        node.notify()
    
    @abstractmethod
    def add_edge(self, edge: Edge) -> None:
        pass
    
    def remove_edge(self, edge: Edge) -> None:
        edge_index = self._edges.index(edge)
        for node_row in self._matrix:
            node_row.pop(edge_index) # remove edge in matrix
        self._edges.remove(edge) # remove edge from list of edges
        edge.notify()
    
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
        self._edges.append(UEdge(node1, node2, weight))
        node1.attach(node2)
        node2.attach(node1)
        for node_index in range(len(self._nodes)):
            self._matrix[node_index].append(int(self._nodes[node_index] in [node1, node2])*weight) # weight value if node a vertice of edge, zero otherwise

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
        self._edges.append(DEdge(node1, node2, weight))
        node1.attach(node2)
        for node_index in range(len(self._nodes)):
            if node := self._nodes[node_index] in [node1, node2]:
                if node == node2:
                    self._matrix[node_index].append(weight) # entering node
                else:
                    self._matrix[node_index].append(-weight) # leaving node
            else:
                self._matrix[node_index].append(0)
