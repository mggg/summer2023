import itertools as it
import networkx as nx
import random
from fractions import Fraction
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from plotting import *


def draw(bg):
    nx.draw(bg, with_labels=True, labels={
            node: "".join(map(str, node)) + "\n" + str(m["ballot_weight"])
            for node, m in bg.nodes(data=True)})
    plt.show()


def ballot_graph(n, k, election=None, remove_k_minus_1=False):
    bg = nx.Graph()
    for num_positions_filled in range(1, k + 1):
        if remove_k_minus_1 and num_positions_filled == k - 1:
            continue
        for ballot in itertools.permutations(range(1, n + 1), num_positions_filled):
            bg.add_node(ballot, ballot_weight=0 if election else 1)
    if election:
        for ballot, ballot_weight in election.items():
            truncated_ballot = ballot[:k]
            if truncated_ballot in bg.nodes():
                bg.add_node(truncated_ballot, ballot_weight=ballot_weight
                            + bg.nodes()[truncated_ballot]["ballot_weight"])
            else:
                raise Exception(f"Unknown ballot found in election: {truncated_ballot}")
    for old_ballot in bg.nodes():
        num_positions_filled = len(old_ballot)
        if num_positions_filled > 1 and not (remove_k_minus_1 and num_positions_filled == k - 1):
            for i in range(num_positions_filled - 1):
                # Add swapping edges.
                new_ballot = list(old_ballot)
                new_ballot[i + 1] = old_ballot[i]
                new_ballot[i] = old_ballot[i + 1]
                bg.add_edge(old_ballot, tuple(new_ballot), edge_weight=1)
            # Add truncation edges.
            if len(old_ballot) == k and remove_k_minus_1:
                bg.add_edge(old_ballot, old_ballot[:-2], edge_weight=1)
            else:
                bg.add_edge(old_ballot, old_ballot[:-1], edge_weight=1)
    return bg


def pickle_distance_matrix(n, k):
    bg = ballot_graph(n, k)
    d = nx.floyd_warshall(bg, weight="edge_weight")
    with open(f"ballot_graph_{n}_{k}", 'wb') as f:
        pickle.dump({v1: {v2: d[v1][v2] for v2 in bg.nodes()} for v1 in bg.nodes()}, f)


def k_means(bg, n, trunc, k, num_repetitions=None):
    if num_repetitions:
        to_return = {}
        for _ in range(num_repetitions):
            centers = frozenset(k_means(bg, n, trunc, k))
            if centers in to_return:
                to_return[centers] += 1
            else:
                to_return[centers] = 1
        return to_return
    with open(f"ballot_graph_{n}_{trunc}", 'rb') as f:
        d = pickle.load(f)
    old_centers = {}
    new_centers = set(random.sample(bg.nodes(), k=k))
    num_iterations = 0
    while old_centers != new_centers:
        old_centers = new_centers
        new_centers = set()
        num_iterations += 1
        #print(num_iterations)
        
        # Compute clusters, dividing equally when there are ties.
        weight_contribution_to_cluster = {center: {} for center in old_centers}
        for node, node_data in bg.nodes(data=True):
            min_distance = min(d[node][center] for center in old_centers)
            divide_among = [center for center in old_centers if d[node][center] == min_distance]
            num_to_divide_among = len(divide_among)
            for center in divide_among:
                weight_contribution_to_cluster[center][node] = \
                        Fraction(node_data["ballot_weight"], num_to_divide_among)
        
        # Define new centers.
        for old_center in old_centers:
            w = weight_contribution_to_cluster[old_center]
            sum_of_distances = {node: 0 for node in bg.nodes()}
            for node in bg.nodes():
                for node_in_cluster, weight_contribution in w.items():
                    sum_of_distances[node] += weight_contribution*d[node][node_in_cluster]
            sorted_nodes = list(bg.nodes())
            random.shuffle(sorted_nodes)
            sorted_nodes = sorted(sorted_nodes, key=lambda node: sum_of_distances[node])
            i = 0
            while sorted_nodes[i] in new_centers:
                i += 1
            new_centers.add(sorted_nodes[i])
    print(f"Convergence after {num_iterations} iterations.")
    return new_centers


def annotate_with_intervals(bg, interval):
    """Directed weighted distances based on Mala's drawing."""
    bdg = bg.to_directed()
    for b1, b2 in bg.edges():
        l1 = len(b1)
        l2 = len(b2)
        if l1 == l2:
            for i in range(l1):
                if b1[i] != b2[i]:
                    break
            w_forward = (interval[b1[i]]/interval[b2[i]])/(2**(i + 1))
            w_backward = (interval[b2[i]]/interval[b1[i]])/(2**(i + 1))
        elif l1 > l2:
            w_forward = w_backward = interval[b2[-1]]/(2**l1)
        else:
            w_forward = w_backward = interval[b1[-1]]/(2**l2)
        bdg.add_edge(b1, b2, edge_weight_using_intervals=w_forward)
        bdg.add_edge(b2, b1, edge_weight_using_intervals=w_backward)
    return bdg


def compute_weighted_distance(bdg, a, b):
    return nx.shortest_path_length(bdg, source=a, target=b, weight="edge_weight_using_intervals")


def interval_aware_clustering(bg, n, trunc, k, discount, num_repetitions=None):
    if num_repetitions:
        to_return = {}
        for _ in range(num_repetitions):
            intervals = frozenset(interval_aware_clustering(bg, n, trunc, k, discount))
            if intervals in to_return:
                to_return[intervals] += 1
            else:
                to_return[intervals] = 1
        return to_return
    with open(f"ballot_graph_{n}_{trunc}", 'rb') as f:
        d = pickle.load(f)
    old_centers = []
    new_centers = list(random.sample(bg.nodes(), k=k))
    new_intervals = [[0] + [1/n for __ in range(n)] for _ in range(k)]
    num_iterations = 0
    while old_centers != new_centers:
        old_centers = new_centers
        new_centers = []
        old_intervals = new_intervals
        bdgs = [annotate_with_intervals(bg, interval) for interval in new_intervals]
        new_intervals = []
        num_iterations += 1
        
        # Compute clusters based on centers and intervals, dividing equally when there are ties.
        weight_contribution_to_cluster = {center: {} for center in old_centers}
        for node, node_data in bg.nodes(data=True):
            weighted_distance = {old_centers[i]: compute_weighted_distance( \
                                                 bdgs[i], old_centers[i], node \
                                                 ) for i in range(k)}
            min_distance = min(weighted_distance[old_centers[i]] for i in range(k))
            divide_among = [center for center in old_centers if weighted_distance[center] \
                            == min_distance]
            num_to_divide_among = len(divide_among)
            for center in divide_among:
                weight_contribution_to_cluster[center][node] = \
                        Fraction(node_data["ballot_weight"], num_to_divide_among)
        
        # Define new centers, same as in k_means.
        for old_center in old_centers:
            w = weight_contribution_to_cluster[old_center]
            sum_of_distances = {node: 0 for node in bg.nodes()}
            for node in bg.nodes():
                for node_in_cluster, weight_contribution in w.items():
                    sum_of_distances[node] += weight_contribution*d[node][node_in_cluster]
            sorted_nodes = list(bg.nodes())
            random.shuffle(sorted_nodes)
            sorted_nodes = sorted(sorted_nodes, key=lambda node: sum_of_distances[node])
            i = 0
            while sorted_nodes[i] in new_centers:
                i += 1
            new_centers.append(sorted_nodes[i])
        
        # Define new intervals from centers.
        for center in new_centers:
            sp = nx.shortest_path_length(bg, source=center, weight="edge_weight")
            interval = [0 for _ in range(n + 1)]
            for ballot, distance in sp.items():
                multiplier = bg.nodes()[ballot]["ballot_weight"] * (discount ** distance)
                ballot_length = len(ballot)
                for i in range(ballot_length):
                    c = ballot[i]
                    interval[c] += (1 - i/ballot_length)*multiplier
            total_weight = sum(interval)
            new_intervals.append([c / total_weight for c in interval])
        
    print(f"Convergence after {num_iterations} iterations.")
    return tuple(map(tuple, new_intervals))


def row_normalize(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)  # Calculate the row sums
    normalized_matrix = matrix / row_sums       # Perform element-wise division
    return normalized_matrix


def prob_matrix_gens_ballot(ballot, matrix, cand_to_mat_ind, full_star=False):
    ncands = len(cand_to_mat_ind.keys())

    total_prob = 1
    total_prob *= matrix[0][cand_to_mat_ind[ballot[0]]]
    for b in range(len(ballot) - 1):
        total_prob *= matrix[cand_to_mat_ind[ballot[b]]][ballot[b + 1]]
    if full_star or len(ballot) < ncands:
        total_prob *= matrix[cand_to_mat_ind[ballot[-1]]][0]
    return total_prob


def gen_transition_probs(election, cand_to_mat_ind, is_normalized=True, full_star=False):
    all_ballots = list(election.keys())
    ncands = len(cand_to_mat_ind.keys())
    mat = np.zeros((ncands + 1, ncands + 1))

    # For following documentation the ballot (ranking) is (C1, C2, C3...)
    for ballot in all_ballots:
        # This adds the weight of the ballot to
        # *,C1 in the matrix
        mat[0][cand_to_mat_ind[ballot[0]]] += election[ballot]
        # This adds the weight of the ballot to
        # Ci,Cj in the matrix
        for b in range(len(ballot) - 1):
            mat[cand_to_mat_ind[ballot[b]]][cand_to_mat_ind[ballot[b+1]]] += election[ballot]
        # This adds the weight of the ballot to
        # CN,* in the matrix
        if full_star or len(ballot) < ncands:
            mat[cand_to_mat_ind[ballot[-1]]][0] += election[ballot]

    if not is_normalized:
        return mat
    norm_mat = row_normalize(mat)
    return norm_mat


def are_numpy_lists_same(lst1, lst2):
    return set([arr.tostring() for arr in lst1]) == set([arr.tostring() for arr in lst2])


def count_2d_array_occurrences(arr_list):
    array_counts = {}

    for arr in arr_list:
        # Convert the 2D NumPy array to a tuple to use it as a dictionary key
        arr_tuple = tuple(map(tuple, arr))

        # Update the dictionary with the count of occurrences of the array
        array_counts[arr_tuple] = array_counts.get(arr_tuple, 0) + 1

    return array_counts


def matrix_cluster_exp(election, n, k, nsims):
    clusters = []
    for _ in range(nsims):
        clusters.extend(matrix_cluster(election=election, n=n, k=k))
    cluster_dict = count_2d_array_occurrences(clusters)
    print(list(cluster_dict.values()))


def matrix_cluster(election, n, k, iter_dest=None, num_repetitions=None):
    if num_repetitions:
        to_return = {}
        for _ in range(num_repetitions):
            intervals = frozenset(tuple(map(tuple, mat)) for mat in matrix_cluster(election, n, k)[0])
            if intervals in to_return:
                to_return[intervals] += 1
            else:
                to_return[intervals] = 1
        return to_return

    iters = 0
    ballot_swaps = []
    all_ballots = list(election.keys())
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    cand_to_mat_ind = {candidates[i]: i + 1 for i in range(len(candidates))}
    nrows = len(candidates) + 1

    # initialize matrix Ms randomly
    old_matrices = [np.zeros((nrows, nrows)) for _ in range(k)]
    new_matrices = [row_normalize(np.random.rand(nrows, nrows)) for _ in range(k)]
    cluster_sizes = [0 for _ in range(k)]

    while not are_numpy_lists_same(old_matrices, new_matrices):
        if iter_dest is not None:
            plot_cluster_matrices(new_matrices,
                                  candidates,
                                  cluster_sizes,
                                  iters,
                                  show=False,
                                  outfile=os.path.join(iter_dest, f"transition_graph_{iters}"))
        old_matrices = new_matrices

        # cluster i is the set of ballots associated with matrix i from old_matrices
        clusters = [dict() for _ in range(k)]
        for ballot, ballot_weight in election.items():
            cluster_fit = {i: prob_matrix_gens_ballot(ballot, old_matrices[i], cand_to_mat_ind) for i in range(k)}
            max_fit = max(cluster_fit.values())
            choose_among = [clusters[i] for i in range(len(clusters)) if cluster_fit[i] == max_fit]
            cluster = np.random.choice(choose_among)
            cluster[ballot] = ballot_weight

        # Generate new matrices that fit the two clusters
        new_matrices = [gen_transition_probs(clusters[i], cand_to_mat_ind) for i in range(len(clusters))]
        cluster_sizes = [sum(cluster.values()) for cluster in clusters]
        iters += 1


    return new_matrices, cluster_sizes, iters


### PARSER ###
def remove_zeros(ballot):
    to_return = []
    for vote in ballot:
        if vote != 0:
            to_return.append(vote)
    return tuple(to_return)


def parse(filename):
    election = {}
    names = []
    numbers = True
    with open(filename, "r") as file:
        for line in file:
            s = line.rstrip("\n").rstrip()
            if numbers:
                ballot = [int(vote) for vote in s.split(" ")]
                num_votes = ballot[0]
                if num_votes == 0:
                    numbers = False
                else:
                    election[remove_zeros(ballot[1:])] = num_votes
            elif "(" not in s:
                return election, names, s.strip("\"")
            else:
                name_parts = s.strip("\"").split(" ")
                first_name = " ".join(name_parts[:-2])
                last_name = name_parts[-2]
                party = name_parts[-1].strip("(").strip(")")
                names.append((first_name, last_name, party))
    raise Exception(f"Error parsing file '{filename}'.")


## August 4
# single-peaked perferences: clusters are defined by all voters in a bloc 
# pefer A or B to C or D, 