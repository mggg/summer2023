from matplotlib import axes
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
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#load in data
folder_path = "/Users/emariedelanuez/summer2023/tda_distance/one_by_one_elim/data_for_one_by_one/one_elimination_15_ro"
def loadin(path):
    json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]
    result = json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
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
    has_negative_values = np.any(dist_matrix < 0)

    # Check for zeros
    has_zeros = np.any(dist_matrix == 0)

    if has_negative_values:
        print("dist_matrix has negative values.")
    else:
        print("dist_matrix does not have negative values.")

    if has_zeros:
        print("dist_matrix has zeros.")
    else:
        print("dist_matrix does not have zeros.")
    
    #there are zeros somewhere


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
            result = np.sqrt(result)/2
            results[i, j] = result
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    scaler = MinMaxScaler()
    normalized_results = scaler.fit_transform(results)
    results_df = pd.DataFrame(normalized_results)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    #print(results_df)  ### note to self there is some weird nan values coming up when i do the square root divided by two stuff. there are no NAN values when i don't do it with the square root. why are those nan values coming up?
    return results
 
resultss= pairwise_comparision(G1, G1)



#try different distances to see if its specific to this function -- to see if you get symmetry 
# sometimes when peopole use distanced they mean similary which is not the same 
#what matrix did i use 
#cite the papers 
#i implemented it in this language 
#how many nodes
#explain the heatmap 


def generate_heatmap_plot(resultss, graphs, cmap='hot', interpolation='nearest', colorbar_label='badjn'):
    
    phrases_to_remove = [ "_", ".json","chicago"] + [str(i) for i in range (21)]
    names_1 = []
    for i in graphs:
        modified_name = i.graph["name"]
        for phrase in phrases_to_remove:
            modified_name = modified_name.replace(phrase, "")
        names_1.append(modified_name)
        
    fig, ax = plt.subplots(figsize=(10, 8)) 

    ax.set_xticks(np.arange(len(resultss)), labels=names_1)
    ax.set_yticks(np.arange(len(resultss)), labels=names_1)
    
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    
    [label.set_fontsize(9) for label in ax.get_xticklabels()]

    scaler = MinMaxScaler()

    normalized_resultss = scaler.fit_transform(resultss)

    heatmap = ax.imshow(normalized_resultss, cmap=cmap, interpolation=interpolation)  

    for i in range(len(resultss)):
        for j in range(len(resultss[i])):
            if i != j:
                ax.text(j, i, f"{normalized_resultss[i, j]:.00002f}", ha="center", va="center", color="blue", fontsize=6)

    colorbar = plt.colorbar(heatmap)
    colorbar.set_label(colorbar_label)
    ax.set_title("odeeeee")
    plt.savefig('/Users/emariedelanuez/summer2023/tda_distance/one_by_one_elim/heatmaps_for_one_by_one/tryme.png')
    plt.show()
    return fig

viz= generate_heatmap_plot(resultss=resultss, graphs = G1, cmap='hot', interpolation='nearest', colorbar_label='try')



