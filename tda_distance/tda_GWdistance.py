import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
import ot
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import itertools
import traceback

#load in data
folder_path = "/Users/emariedelanuez/summer2023/tda_distance/toy_tests_dbscan"
def loadin(path):
    json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]
    result = json_files.sort()
    graphs = []
    for file in json_files:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, "r") as f:
                json_data = json.load(f)
                G = json_graph.adjacency_graph(json_data["graph"])
                G.graph["name"] = file_path.split("/")[-1]
                graphs.append(G)
                
        except FileNotFoundError:
            print(f"JSON file '{file}' not found in the folder.")
        except json.JSONDecodeError as e:
            print(f"Error while decoding JSON in file '{file}': {e}")
        except IOError as e:
            print(f"Error while reading the file '{file}': {e}")
    

    """""
    might not be the most well executed and should get touched up 
    if you want to compare two completely different families of graphs 
    
    """
    return graphs #, obj_2, graphs_2
G1 = loadin(folder_path)



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
        weight =  abs(G.nodes[edge[0]]["mean_filter_val"] - G.nodes[edge[1]]["mean_filter_val"])
        
        if weight == 0:
            weight = np.finfo(np.float64).eps # force positive-definite
        G.edges[edge]["weight"] = weight
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


def pairwise_comparision(graphs_1, graphs_2):
    decorated_graphs1 = [decorate_graph(graph) for graph in graphs_1]
    decorated_graphs2 = [decorate_graph(graph) for graph in graphs_2]

    results = np.empty((len(graphs_1), len(graphs_2))) 
    for i, decorated_G1 in tqdm(enumerate(decorated_graphs1), desc = "Pairwise distances outer loop"): 
        for j, decorated_G2 in tqdm(enumerate(decorated_graphs2), desc = "Pairwise distances inner loop", leave=False):
            result = gw_dist(decorated_G1, decorated_G2)
            results[i, j] = result  

    return results
 

resultss = pairwise_comparision(G1, G1)

def generate_heatmap_plot(resultss, graphs, cmap='hot', interpolation='nearest', label='Data Values'):
    names_1 = []
    for i in graphs:
        names_1.append(i.graph["name"].replace("_","").strip("toy.json"))
    print(names_1)

    fig, ax = plt.subplots()  

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(resultss)), labels=names_1)
    ax.set_yticks(np.arange(len(resultss)), labels=names_1)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    heatmap = ax.imshow(resultss, cmap=cmap, interpolation=interpolation) 
    colorbar = plt.colorbar(heatmap)
    colorbar.set_label(label)
    ax.set_title("toy dbscan")
    plt.savefig('/Users/emariedelanuez/summer2023/tda_distance/heatmaps/toytests_dbscan.png')
    plt.show()
    return fig

viz= generate_heatmap_plot(resultss=resultss, graphs = G1, cmap='hot', interpolation='nearest', label='Data Values')
print(viz)
""""



im = ax.imshow(resultss)



# Loop over data dimensions and create text annotations.
for i in range(len(G1)):
    for j in range(len(G1)):
        text = ax.text(j, i, G1[i, j],
                       ha="center", va="center", color="w")
        


ax.set_title("whatever")
fig.tight_layout()
plt.show()

"""

"""


# Assuming g1 and g2 are lists of graphs obtained from loadin()

def pairwise_comparision(graphs_1, graphs_2):
    decorated_1 = [ decorate_graph(graph) for graph in graphs_1 ]
    decorated_2 = [ decorate_graph(graph) for graph in graphs_2 ]
    
    for graph_i in decorated_1:
        for node, data in graph_i.nodes(data=True):
            for attr_name, attr_value in data.items():
                if attr_value == 0:
                    print(f"Node {node} in graph_i has attribute '{attr_name}' with a value of zero.")
    return decorated_1, decorated_2

check = pairwise_comparision(G1, G2)
print(check)
  

#
#obj_1 = json.load(open("/Users/emariedelanuez/kepler-mapper/output for implementation/chicago_vapandsalary_jordans_version.json"))
#G1 = json_graph.adjacency_graph(obj_1["graph"])
#obj_2 = json.load(open("/Users/emariedelanuez/kepler-mapper/output for implementation/chicago_justwvapsalary.json"))
#G2 = json_graph.adjacency_graph(obj_2["graph"])

"""