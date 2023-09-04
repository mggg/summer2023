import itertools
import networkx as nx
from gerrychain.random import random
import math
from matplotlib import pyplot as plt


def degree_biased_walk_tree(graph, pop_col, start_vertex):
    g1 = graph.copy()
    spine = set()
    spine_edges = set()
    current_vertex = start_vertex
    for first_round in [True, False]:
        while current_vertex is not None:
            neighbors = list(g1.neighbors(current_vertex))
            min_priority = 99999999
            min_priority_nodes = []
            for n in neighbors:
                degree = g1.degree(n)
                priority = 99999999 if degree == 1 else abs(degree - 2)
                if priority < min_priority:
                    min_priority = priority
                    min_priority_nodes = [n]
                elif priority == min_priority:
                    min_priority_nodes.append(n)
            #print(f"Min priority: {min_priority}")
            spine.add(current_vertex)
            g1.remove_node(current_vertex)
            if min_priority_nodes == []:
                current_vertex = None
            else:
                next_vertex = random.choice(min_priority_nodes)
                spine_edges.add((current_vertex, next_vertex))
                current_vertex = next_vertex
        if first_round:
            neighbors = list(set(graph.neighbors(start_vertex)) - spine)
            if len(neighbors) > 0:
                current_vertex = random.choice(neighbors)
    return finish(graph, pop_col, start_vertex, spine_edges)


def cycle_bite_walk_tree(graph, pop_col, cutoff, start_vertex, plot=0):
    num_rounds_since_last_improvement = 0
    p = [start_vertex]
    s = {start_vertex}
    v = start_vertex
    while num_rounds_since_last_improvement < cutoff:
        if num_rounds_since_last_improvement >= cutoff - plot:  # For grids only.
            side_length = int(math.sqrt(len(graph)))
            board = [[0 for __ in range(side_length)] for _ in range(side_length)]
            for i in range(len(p)):
                a, b = p[i]
                board[a][b] = i
            im = plt.imshow(board, interpolation="nearest")
            plt.axis('off')
            im.set_data(board)
            plt.show()
        num_rounds_since_last_improvement += 1
        neighbors = set(graph.neighbors(v))
        neighbors_in_spine, neighbors_not_in_spine = neighbors.intersection(s), neighbors - s
        if len(neighbors_not_in_spine) == 0:
            if len(neighbors_in_spine) > 1:
                neighbors_in_spine.remove(p[-2])
            v_in_s = random.choice(list(neighbors_in_spine))
            new_path = []
            while p[-1] != v_in_s:
                new_path.append(p.pop())
            p += new_path
            v = p[-1]
        else:
            v = random.choice(list(neighbors_not_in_spine))
            p.append(v)
            s.add(v)
            #print(num_rounds_since_last_improvement)
            num_rounds_since_last_improvement = 0
    return finish(graph, pop_col, start_vertex, [(p[i], p[i + 1]) for i in range(len(p) - 1)])


def finish(graph, pop_col, start_vertex, spine_edges):
    g2 = graph.copy()
    for edge in graph.edges:
        e1, e2 = edge
        if (e1, e2) in spine_edges or (e2, e1) in spine_edges:
            g2.edges[edge]["weight"] = 0
        else:
            g2.edges[edge]["weight"] = graph.nodes[e1][pop_col] + graph.nodes[e2][pop_col]
    shortest_paths = nx.shortest_path(g2, source=start_vertex, weight="weight")
    spanning_tree = nx.Graph()
    spanning_tree.add_nodes_from(graph.nodes(data=True))
    print(f"Spine length: {len(spine_edges)}")
    for p in shortest_paths.values():
        for i in range(len(p) - 1):
            spanning_tree.add_edge(p[i], p[i + 1])
    return spanning_tree

