import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
import ot

#load in data
obj_1 = json.load(open("/Users/emariedelanuez/kepler-mapper/output for implementation/chicago_vapandsalary_jordans_version.json"))
G1 = json_graph.adjacency_graph(obj_1["graph"])
obj_2 = json.load(open("/Users/emariedelanuez/kepler-mapper/output for implementation/chicago_justwvapsalary.json"))
G2 = json_graph.adjacency_graph(obj_2["graph"])



def sum_node_members(G):
    total_sum = 0
    for node, data in G.nodes(data=True):
        total_sum += len(data["membership"])

    G.graph["total_node_membership"] = total_sum
    return G


def decorate_nodes(G):
    for node in G.nodes():
        G.nodes[node]["weight"] = len(G.nodes[node]["membership"])/G.graph["total_node_membership"]
        G.nodes[node]["filter_len"] = len(G.nodes[node]["filter_vals"])
        G.nodes[node]["mean_filter_val"] = np.array(G.nodes[node]["filter_vals"]).sum() / G.nodes[node]["filter_len"]
    return G


def decorate_edges(G):
    edges = G.edges()
    for edge in edges:
        G.edges[edge]["weight"] = abs(G.nodes[edge[0]]["mean_filter_val"] - G.nodes[edge[1]]["mean_filter_val"])

    return G
def distance_matrix(G):
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dist_matrix = np.full((len(G.nodes), len(G.nodes)), -np.inf)

    for source, paths in shortest_paths.items():
        source_index = list(G.nodes).index(source)

        for target, length in paths.items():
            target_index = list(G.nodes).index(target)
            dist_matrix[source_index, target_index] = length

    dist_matrix[dist_matrix == -np.inf] = dist_matrix.max()
    G.graph["dist_matrix"] = dist_matrix
    return G
def decorate_graph(G):
    G = sum_node_members(G)
    G = decorate_nodes(G)
    G = decorate_edges(G)
    G = distance_matrix(G)
    return G

def gw_dist(G1, G2):

    dist = ot.gromov.gromov_wasserstein2(G1.graph["dist_matrix"],G2.graph ["dist_matrix"],
                                         list(dict(G1.nodes(data="weight")).values()),
                                         list(dict(G2.nodes(data="weight")).values())
                                         )
    return dist

result = gw_dist(decorate_graph(G1),decorate_graph(G2))

print(result)

