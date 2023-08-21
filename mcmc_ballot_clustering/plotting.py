from clustering import *
from tqdm import tqdm
from intervals import *
import numpy as np 
import matplotlib.pyplot as plt


has_plot = False


def visualize_intervals_borda(filename, bg_cutoff, num_repetitions, radius, discount):
    global has_plot
    has_plot = True
    election, names, location = parse(filename)
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, bg_cutoff, election)
    centers_dict = k_means(bg, n, bg_cutoff, 2, num_repetitions=num_repetitions)
    a = sorted([(-occurrences, centers) for centers, occurrences in centers_dict.items()])
    num_plots = len(a)
    extra = num_plots == 1
    if extra:
        num_plots = 2
    fig, axs = plt.subplots(num_plots)
    for i in range(num_plots):
        occurrences, centers = a[i]
        intervals = [get_interval_borda( \
                    n, bg, center, radius=radius, discount=discount) \
                    for center in centers]
        X_axis = np.arange(n + 1)
        if intervals[0][1] >= intervals[1][1]:
            i1 = 0
            i2 = 1
        else:
            i1 = 1
            i2 = 0
        axs[i].bar(X_axis - 0.2, intervals[i1], 0.4)
        axs[i].bar(X_axis + 0.2, intervals[i2], 0.4)
        axs[i].text(0, max(intervals[0] + intervals[1])/2, str(-occurrences), ha="right")
        if extra:
            return


def visualize_intervals_iac(filename, bg_cutoff, num_repetitions, discount):
    global has_plot
    has_plot = True
    election, names, location = parse(filename)
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    bg = ballot_graph(n, bg_cutoff, election)
    intervals_dict = interval_aware_clustering(bg, n, bg_cutoff, 2, discount, num_repetitions=num_repetitions)
    a = sorted([(-occurrences, tuple(intervals)) for intervals, occurrences in intervals_dict.items()])
    num_plots = len(a)
    extra = num_plots == 1
    if extra:
        num_plots = 2
    fig, axs = plt.subplots(num_plots)
    for i in range(num_plots):
        occurrences, intervals = a[i]
        X_axis = np.arange(n + 1)
        if intervals[0][1] >= intervals[1][1]:
            i1 = 0
            i2 = 1
        else:
            i1 = 1
            i2 = 0
        axs[i].bar(X_axis - 0.2, intervals[i1], 0.4)
        axs[i].bar(X_axis + 0.2, intervals[i2], 0.4)
        axs[i].text(0, max(intervals[0] + intervals[1])/2, str(-occurrences), ha="right")
        if extra:
            return


def plot_cluster_matrices(adjacency_matrices, candidates, cluster_sizes, iter=None, show=False, outfile=None):
    # Set up fig
    fig, axes = plt.subplots(1, len(adjacency_matrices), figsize=(6 * len(adjacency_matrices), 6))
    for a, adj_matrix in enumerate(adjacency_matrices):
        # Handle single vs multiple axes
        ax = axes
        if len(adjacency_matrices) > 1:
            ax = axes[a]

        ax.axis('equal')
        if iter is None:
            ax.set_title(f"Transition Cluster {a+1}")
        else:
            ax.set_title(f"Transition Cluster {a + 1}, iter {iter}")
        ax.set_xlabel(f"Cluster size: {cluster_sizes[a]}")

        # Create a directed graph
        G = nx.DiGraph()

        # Get the number of nodes in the graph
        num_nodes = adj_matrix.shape[0]

        # Add nodes to the graph
        G.add_nodes_from(range(num_nodes))

        # Add edges to the graph with weights
        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = adj_matrix[i, j]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)

        pos = nx.circular_layout(G)

        # Networkx plotting
        nx.draw_networkx(G,
                         pos,
                         with_labels=False,
                         node_size=300,
                         node_color='skyblue',
                         edgelist=list(),
                         ax=ax)
        node_labels = ["*"] + candidates
        labels = {i: label for i, label in enumerate(node_labels)}
        nx.draw_networkx_labels(G,
                                pos,
                                labels,
                                font_size=12,
                                font_weight='bold',
                                ax=ax)
        for edge in G.edges(data='weight'):
            color = rgb_to_hex(tuple(np.repeat(int(255 * (1 - edge[2])), 3)))
            nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist=[edge],
                                   edge_color=color,
                                   width=2 * edge[2],
                                   connectionstyle='arc3, rad = 0.1',
                                   ax=ax)
        # edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
        # nx.draw_networkx_edge_labels(G,
        #                              pos,
        #                              edge_labels=edge_labels,
        #                              label_pos=0.3,
        #                              verticalalignment='top',
        #                              font_size=6)

    # Out stuff
    if show: plt.show()
    if outfile is not None: plt.savefig(outfile)
    plt.close(fig)


def visualize_intervals_markov(filename, num_repetitions):
    global has_plot
    has_plot = True
    election, names, location = parse(filename)
    n = len(names)
    for i in range(n):
        print(f"{i + 1}: {names[i]}")
    matrices_dict = matrix_cluster(election, n, 2, num_repetitions=num_repetitions)
    a = sorted([(-occurrences, matrices) for matrices, occurrences in matrices_dict.items()])
    num_plots = len(a)
    extra = num_plots == 1
    if extra:
        num_plots = 2
    fig, axs = plt.subplots(num_plots)
    for i in range(num_plots):
        occurrences, matrices = a[i]
        intervals = [[matrix[0][i] for i in range(n + 1)] for matrix in matrices]
        X_axis = np.arange(n + 1)
        if intervals[0][1] >= intervals[1][1]:
            i1 = 0
            i2 = 1
        else:
            i1 = 1
            i2 = 0
        axs[i].bar(X_axis - 0.2, intervals[i1], 0.4)
        axs[i].bar(X_axis + 0.2, intervals[i2], 0.4)
        axs[i].text(0, max(intervals[0] + intervals[1])/2, str(-occurrences), ha="right")
        if extra:
            return


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
