import json
import random
from collections import Counter
import heapq

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.constraints import contiguous
from gerrychain.proposals import recom
from functools import partial
import pandas
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.updaters import cut_edges


def find_star_partition(graph: Graph, num_districts: int, updaters) -> Partition:
    """Partition the graph into `num_districts - 1` single-block districts and one district containing all other blocks.
    In particular, populations will not be balanced, but districts will be contiguous.
    """

    assignment = {node: 0 for node in graph.nodes}

    for district in range(1, num_districts):
        for node in graph.nodes:
            if assignment[node] != 0:
                continue
            assignment[node] = district  # try turning node into single district
            if contiguous(Partition(graph, assignment)):  # contiguous_bfs would be *really* slow here
                break
            else:
                assignment[node] = 0
        if assignment[node] == 0:
            raise ValueError(f"Failed to initialize the {district}th single-block district due to contiguity concerns. "
                             f"Either your number of districts is really low, or I’m using the contiguity check wrong.")

    return Partition(graph, assignment, updaters)


def graph_voronoi(graph: Graph, num_districts: int, updaters) -> Partition:
    while True:
        starting_nodes = random.choices(list(graph.nodes), [graph.nodes[node][pop_col] for node in graph.nodes], k=num_districts)
        if len(set(starting_nodes)) == num_districts:
            break
    assignment = {starting_node: district for district, starting_node in enumerate(starting_nodes)}
    district_priority = [(graph.nodes[starting_node][pop_col], district) for district, starting_node in enumerate(starting_nodes)]
    heapq.heapify(district_priority)
    priority_queues = []
    for district, starting_node in enumerate(starting_nodes):
        distances = []
        for neighbor in graph[starting_node]:
            #distances.append((graph.nodes[starting_node][pop_col] + graph.nodes[neighbor][pop_col], neighbor))
            distances.append((1, neighbor))
        heapq.heapify(distances)
        priority_queues.append(distances)

    num_nodes = len(graph.nodes)

    while len(assignment) < num_nodes:
        old_pop, district = heapq.heappop(district_priority)  # take currently smallest district
        priorities = priority_queues[district]
        while len(priorities) > 0:
            dist, neighbor = heapq.heappop(priorities)
            if neighbor in assignment:
                continue
            assignment[neighbor] = district
            heapq.heappush(district_priority, (old_pop + graph.nodes[neighbor][pop_col], district))
            for neighbor2 in graph[neighbor]:
                if neighbor2 in assignment:
                    continue
                #heapq.heappush(priorities, (dist + graph.nodes[neighbor][pop_col] + graph.nodes[neighbor2][pop_col], neighbor2))
                heapq.heappush(priorities, (dist + 1, neighbor2))
            break

    return Partition(graph, assignment, updaters)


def weight_uniform(boundary, large_pop, small_pop):
    return 1

def weight_boundary(boundary, large_pop, small_pop):
    return boundary

def weight_population_ratio(boundary, large_pop, small_pop):
    return large_pop / (small_pop + 1)  # add 1 to prevent division by zero

def weight_population_diff(boundary, large_pop, small_pop):
    return large_pop - small_pop + 100


#def weight_mix(boundary, large_pop, small_pop):
#    return (large_size - small_size) + avg_pop * (1 + count / max_boundary)


def district_boundary_distribution(partition, num_districts, weight_function):
    # don't want to bias merging districts with large boundaries for now, because that might get in the way of
    # eliminating single-block districts.

    district_boundary_counter = Counter()
    for block1, block2 in partition["cut_edges"]:
        district1 = partition.assignment.mapping[block1]
        district2 = partition.assignment.mapping[block2]
        district_boundary = tuple(sorted((district1, district2)))
        district_boundary_counter[district_boundary] += 1

    avg_pop = sum(partition["population"].values()) / num_districts
    _, max_boundary = district_boundary_counter.most_common(1)[0]

    district_boundary_weights = {}
    for (district1, district2), count in district_boundary_counter.items():
        if len(partition.parts[district1]) == len(partition.parts[district2]) == 1:
            # Recom throws a Python exception when trying to combine two single-block districts because all nodes
            # have degree one.
            continue
        small_size, large_size = sorted((partition["population"][district1], partition["population"][district2]))
        district_boundary_weights[(district1, district2)] = weight_function(count, large_size, small_size) # count**(large_size / small_size)

    normalization = sum(district_boundary_weights.values())
    district_boundary_weights = {db: w / normalization for db, w in district_boundary_weights.items()}
    return district_boundary_weights


def equalize_populations(graph, num_districts, weight_function):
    my_updaters = {"population": updaters.Tally(pop_col, alias="population"),
                   "cut_edges": cut_edges}
    initial_partition = find_star_partition(graph, num_districts, my_updaters)
    print(initial_partition)

    partition = initial_partition
    save_partitions = {0: partition}

    population_log = [sorted(partition["population"].values())]

    for iteration in range(1, 5001):
        print(iteration, np.percentile(list(partition["population"].values()), q=[0, 25, 50, 75, 100]))

        dist = district_boundary_distribution(partition, num_districts, weight_function)
        # adapted from recom code
        district1, district2 = random.choices(list(dist.keys()), weights=list(dist.values()))[0]
        print("merge:", (district1, district2), "populations:", (partition["population"][district1], partition["population"][district2]))
        subgraph = partition.graph.subgraph(
            partition.parts[district1] | partition.parts[district2]
        )

        try:
            district2nodes = bipartition_tree(
                subgraph.graph,
                pop_col=pop_col,
                pop_target=(partition["population"][district1] + partition["population"][district2]) / 2,
                epsilon=.05,
                node_repeats=1,
                max_attempts=10
            )
        except RuntimeError:
            print("Couldn’t recom.")
            continue

        flips = {node: district1 for node in subgraph.graph.nodes}
        for node in district2nodes:
            flips[node] = district2
        partition = partition.flip(flips)
        print("new populations: ", (partition["population"][district1], partition["population"][district2]))
        population_log.append(sorted(partition["population"].values()))

        if iteration % 100 == 0:
            save_partitions[iteration] = partition

    return save_partitions, population_log


def plot_metric(population_logs, metric_func, metric_name):
    plt.figure()
    for heuristic, distributions in population_logs.items():
        metric_values = [metric_func(distribution) for distribution in distributions]
        plt.plot(range(len(distributions)), metric_values, label=heuristic)

    plt.xlabel('Time Steps')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'{metric_name} over Time Steps')
    plt.savefig(f'{metric_name}_plot.pdf')
    plt.close()


# Define the metrics to be used
def min_size(distribution):
    return min(distribution)


def max_size(distribution):
    return max(distribution)


def median_size(distribution):
    return np.median(distribution)


def std_dev_size(distribution):
    return np.std(distribution)


def plot_min_max(population_logs):
    plt.figure()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for (heuristic, distributions), color in zip(population_logs.items(), colors):
        metric_values = [min_size(distribution) for distribution in distributions]
        plt.plot(range(len(distributions)), metric_values, color=color, linestyle="-", label=heuristic)
        metric_values = [max_size(distribution) for distribution in distributions]
        plt.plot(range(len(distributions)), metric_values, color=color, linestyle="--", label=None)

    plt.xlabel('Time Steps')
    plt.ylabel("Population")
    plt.yscale('log')
    plt.legend()
    plt.title(f'Minimum and maximum population over Time Steps')
    plt.savefig(f'minmax_plot.pdf')
    plt.close()



pop_col = "TOTPOP20"

if __name__ == "__main__":
    #graph = Graph.from_json("../PA_VTDs.json")
    graph = Graph.from_json("ga_vtd_dual.json")
    num_districts = 20

    geography_df = gpd.read_file("ga_vtd.shp")

    for iter in range(10):
        partition = graph_voronoi(graph, num_districts, {"population": updaters.Tally(pop_col, alias="population"),
                       "cut_edges": cut_edges})
        print(np.percentile(list(partition["population"].values()), q=[0, 25, 50, 75, 100]))
        col_name = "voronoi_unitsteps_" + str(iter)
        geography_df[col_name] = geography_df["GEOCODE"].map(dict(partition.assignment))
        geography_df.plot(column=col_name, cmap="tab20")
        plt.savefig(col_name + ".png")

    from sys import exit
    exit(0)

    # new ideas:
    # - just getting every pair within 5% of each other isn't enough to get every district within 5% of target
    # - one way to overcome: reduce the threshold when splitting pairs. but that’s still tricky, exponential loss over
    #   path from small to large district
    # - maybe at some point switch to different target?
    # - other cute idea: build a flow network of districts that can push capacity into each other. find augmenting flow,
    #   i.e., ones that increase the size of districts below target and decrease the size of districts above target.
    #   on quick glance, not how flows were used before. the tricky part is defining the region where district 1 can
    #   safely move population to district 2 without interfering with connectedness or other transfers.

    weights = {"pop ratio": weight_population_ratio, "pop diff": weight_population_diff, "uniform": weight_uniform,
               "boundary": weight_boundary}
    population_logs = {}

    for weight_name, weight_function in weights.items():
        saved_partitions, population_logs[weight_name] = equalize_populations(graph, num_districts, weight_function)
        with open("./pop_balancing_logs.json", "w") as file:
            json.dump(population_logs, file)

        # Generate the plots
        plot_min_max(population_logs)
        plot_metric(population_logs, min_size, 'Minimum Size')
        plot_metric(population_logs, max_size, 'Maximum Size')
        plot_metric(population_logs, median_size, 'Median Size')
        plot_metric(population_logs, std_dev_size, 'Standard Deviation')

        for iteration, partition in saved_partitions.items():
            col_name = weight_name + str(iteration)
            geography_df[col_name] = geography_df["GEOID10"].map(dict(partition.assignment))
            geography_df.plot(column=col_name, cmap="tab20")
            plt.savefig(col_name + ".png")

    with open("./pop_balancing_logs.json", "r") as file:
        population_logs = json.load(file)

    # Generate the plots
    plot_min_max(population_logs)
    plot_metric(population_logs, min_size, 'Minimum Size')
    plot_metric(population_logs, max_size, 'Maximum Size')
    plot_metric(population_logs, median_size, 'Median Size')
    plot_metric(population_logs, std_dev_size, 'Standard Deviation')
