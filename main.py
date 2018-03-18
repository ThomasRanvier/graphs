import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph():
    def __init__(self):
        self.edges = []
        self.max = 0;

    def add_oriented_edge(self, id1, id2, weight):
        add = True
        if self.max < id1:
            self.max = id1
        if self.max < id2:
            self.max = id2
        for edge in self.edges:
            if edge.node1 == id1 and edge.node2 == id2:
                add = False
        if add:
            self.edges.append(Edge(id1, id2, weight))
        else:
            print 'Node doesn\'t exist or edge already exists'

    def add_double_edge(self, id1, id2, weight):
        self.add_oriented_edge(id1, id2, weight)
        self.add_oriented_edge(id2, id1, weight)

    def get_adj_matrix(self):
        mat = np.zeros(shape=(self.max,self.max))
        for edge in self.edges:
                mat[edge.node1 - 1][edge.node2 - 1] = 1
        return mat

    def draw(self):
        graph = nx.DiGraph()
        for edge in self.edges:
            graph.add_edge(edge.node1, edge.node2, weight=edge.weight)
        pos=nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        labels = nx.get_edge_attributes(graph, 'weight')
        #ed,weights = zip(*labels.items())
        nx.draw_networkx(graph, pos, node_color='#A0CBE2')#, edge_color=weights,
        #width=4, edge_cmap=plt.cm.winter)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.show()

    def __str__(self):
        str = '\n'
        for edge in self.edges:
            str += '{} -> {} : {}\n'.format(edge.node1,
            edge.node2, edge.weight)
        return str

    def dijkstra(self, source):
        visited = {source: 0}
        path = {}
        nodes = []

        for n in range(1, self.max + 1):
            nodes.append(n)

        while nodes:
            min_node = None
            for node in nodes:
                if node in visited:
                    if min_node is None:
                        min_node = node
                    elif visited[node] < visited[min_node]:
                        min_node = node

            if min_node is None:
                break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in self.edges:
                if edge.node1 == min_node and edge.node2 in nodes:
                    weight = current_weight + edge.weight
                    if edge.node2 not in visited or weight < visited[edge.node2]:
                        visited[edge.node2] = weight
                        path[edge.node2] = min_node

        return path, visited

    def delete_edge(self, node1, node2):
        new_max = 0
        count = 0
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2:
                del self.edges[count]
                break
            count += 1
        for edge in self.edges:
            if edge.node1 > new_max:
                new_max = edge.node1
            if edge.node2 > new_max:
                new_max = edge.node2
        self.max = new_max

    def prim(self):
        tree = Graph()
        edges_cp = []
        for edge in self.edges:
            edges_cp.append([edge.node1, edge.node2, edge.weight])
        edges_cp = sorted(edges_cp,key=lambda item: item[2])
        tree.add_oriented_edge(edges_cp[0][0], edges_cp[0][1], edges_cp[0][2])
        nodes = [edges_cp[0][0], edges_cp[0][1]]
        del edges_cp[0]
        while len(nodes) < self.max:
            count = 0
            for u,v,w in edges_cp:
                if u in nodes and v not in nodes:
                    tree.add_oriented_edge(u, v, w)
                    nodes.append(v)
                    del edges_cp[count]
                    break
                count += 1
        return tree

    def get_neighbors(self, source):
        neighbors = set()
        for edge in self.edges:
            if edge.node1 == source:
                neighbors.add(edge.node2)
            if edge.node2 == source:
                neighbors.add(edge.node1)
        return neighbors

    def get_connected_components(self, source):
        tree = Graph()
        nodes = set()
        not_seen = []
        not_seen.append(source)

        while not_seen:
            node = not_seen[0]
            del not_seen[0]
            nodes.add(node)
            neighbors = self.get_neighbors(node)
            for neigh in neighbors:
                if neigh not in nodes and neigh not in not_seen:
                    not_seen.append(neigh)

        for edge in self.edges:
            if edge.node1 in nodes and edge.node2 in nodes:
                tree.add_oriented_edge(edge.node1, edge.node2, edge.weight)

        return tree

class Edge():
    def __init__(self, node1, node2, weight):
        self.weight = weight
        self.node1 = node1
        self.node2 = node2

def display_graph_from_matrix(matrix):
    graph = nx.from_numpy_matrix(matrix, create_using=nx.MultiDiGraph())
    nx.draw(graph, with_labels=True)
    plt.show()

if __name__ == '__main__':
    g = Graph()

    g.add_double_edge(1, 4, 0.5)
    g.add_double_edge(5, 6, 6)
    g.add_double_edge(3, 7, 1)
    g.add_double_edge(3, 4, 3.5)
    g.add_double_edge(1, 2, 4)
    g.add_double_edge(2, 3, 1.5)
    g.add_double_edge(1, 7, 3)
    g.add_double_edge(1, 3, 4.5)
    g.add_double_edge(6, 7, 5)
    g.add_double_edge(1, 6, 5.5)
    g.add_double_edge(4, 6, 6)
    g.add_double_edge(4, 7, 2)
    g.add_double_edge(3, 6, 7)
    g.add_double_edge(2, 4, 7.5)
    g.add_double_edge(8, 9, 10)

    adj_mat = g.get_adj_matrix()
    print 'Matrice d\'adjacences : \n{}\n'.format(adj_mat)
    path, weights = g.dijkstra(2)
    print 'Dijkstra(2) : \n - path : {} \n - weights : {}\n'.format(path,
    weights)

    g.draw()

    connected = g.get_connected_components(2)
    print 'Composante connexe : {}'.format(connected)
    connected.draw()

    connected.prim().draw()
