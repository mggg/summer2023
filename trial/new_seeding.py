import geopandas as gpd
from gerrychain import Graph, Partition, MarkovChain, Election
from gerrychain.constraints import within_percent_of_ideal_population
from gerrychain.accept import always_accept
from functools import partial
from gerrychain.proposals import recom
from gerrychain.updaters import Tally
from gerrychain.tree import recursive_seed_part, bipartition_tree
import time
from gerrychain.random import random
import networkx as nx
from tree_partition import *
from pathy import *


s = int(time.time())
random.seed(s)
print(f"RNG seed: {s}")
pop_col = "TOTPOP20"

graph = Graph.from_json("ga_vtd_dual.json") #parses the dual graph(math graph) and turns it into a gerry chain object (network x )

total_pop = sum([graph.nodes[n][pop_col] for n in graph.nodes])
#population of a node is with pop_col
#pop tol is how uneve the population is the districts can be 
#node== vertex of a grapgh geographic region
#return == {12 (id node): (o^th district)}
#to do seeding is to do line 32  and give it a lot of parameter 
#recursive seed part is the newer version of recursive tree part 

def seed_with_rsp(num_districts, pop_tol):
    ideal_pop = total_pop / num_districts
    initial_assignment = recursive_seed_part(graph,
                                             range(num_districts),
                                             ideal_pop,
                                             pop_col,
                                             pop_tol,
                                             method=bipartition_tree)
    return initial_assignment


def seed_with_one_shot_random_tree(num_districts, pop_tol):
    ideal_pop = total_pop / num_districts
    min_pop = int(ideal_pop - (pop_tol*ideal_pop)) + 1
    max_pop = int(ideal_pop + (pop_tol*ideal_pop)) - 1
    initial_assignment = None
    while initial_assignment is None:
        t = random_spanning_tree(graph)
        initial_assignment, num_partitions = sample_uniform_partition_of_tree(
            t, num_districts, min_pop, max_pop, pop_col)
    return initial_assignment


def seed_with_one_shot_dfs_tree(num_districts, pop_tol):
    ideal_pop = total_pop / num_districts
    min_pop = int(ideal_pop - (pop_tol*ideal_pop)) + 1
    max_pop = int(ideal_pop + (pop_tol*ideal_pop)) - 1
    initial_assignment = None
    while initial_assignment is None:
        root = random.choice(list(graph.nodes))    
        # TODO: randomize edge traversal order!
        t = nx.dfs_tree(graph, source=root).to_undirected()
        for n in list(graph.nodes):
            t.nodes[n][pop_col] = graph.nodes[n][pop_col]
        initial_assignment, num_partitions = sample_uniform_partition_of_tree(
            t, num_districts, min_pop, max_pop, pop_col)
    return initial_assignment


def seed_with_degree_biased_walk(num_districts, pop_tol):
    ideal_pop = total_pop / num_districts
    min_pop = int(ideal_pop - (pop_tol*ideal_pop)) + 1
    max_pop = int(ideal_pop + (pop_tol*ideal_pop)) - 1
    start_vertices = []
    for n in graph.nodes():
        if graph.degree(n) == 4:
            start_vertices.append(n)
    initial_assignment = None
    while initial_assignment is None:
        #print("...")
        start_vertex = random.choice(start_vertices)
        t = degree_biased_walk_tree(graph, pop_col, start_vertex)
        initial_assignment, num_partitions = sample_uniform_partition_of_tree(
            t, num_districts, min_pop, max_pop, pop_col)
    return initial_assignment


def seed_with_cycle_bite_walk(num_districts, pop_tol, cutoff, plot=0):
    ideal_pop = total_pop / num_districts
    min_pop = int(ideal_pop - (pop_tol*ideal_pop)) + 1
    max_pop = int(ideal_pop + (pop_tol*ideal_pop)) - 1
    start_vertices = []
    for n in graph.nodes():
        if graph.degree(n) == 4:
            start_vertices.append(n)
    initial_assignment = None
    while initial_assignment is None:
        print("...")
        start_vertex = random.choice(start_vertices)
        t = cycle_bite_walk_tree(graph, pop_col, cutoff, start_vertex, plot=plot)
        initial_assignment, num_partitions = sample_uniform_partition_of_tree(
            t, num_districts, min_pop, max_pop, pop_col)
    return initial_assignment


test_num = 1

if test_num == 1:
    seed = seed_with_rsp(5, 0.05)
    print(seed)
    print(len(seed))
elif test_num == 2:  # Recursive seed part once got stuck at 25 districts.
    t_start = time.time()
    for num_districts in range(2, 101):
        seed_with_rsp(num_districts, 0.05)
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 3:  # RSP makes it but starts taking ~5 seconds per plan towards the end.
    t_start = time.time()
    for num_districts in range(2, 51):
        seed_with_rsp(num_districts, 0.10)
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 4:  # Test one shot DP method, compare to test 2.
    t_start = time.time()
    for num_districts in range(2, 51):
        initial_assignment = seed_with_one_shot_random_tree(num_districts, .05)
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        #print(initial_assignment)
        t_start = t
elif test_num == 5:
    populations = sorted(graph.nodes[node]["TOTPOP20"] for node in graph.nodes)
    print(populations)
    print(total_pop)
elif test_num == 6:
    populations = sorted(graph.degree(node) for node in graph.nodes)
    print(populations)
    print(total_pop)
elif test_num == 7:
    print(len(graph.nodes()))
    print(len(nx.maximal_matching(graph)))
elif test_num == 8:
    t = nx.path_graph(14)
    for v in t:
        t.nodes[v]["pop"] = 1
    partition, num_partitions = sample_uniform_partition_of_tree(
        t,
        num_parts=4,
        min_pop=3,
        max_pop=4,
        pop_col="pop"
    )
    print(num_partitions)
    print(partition)
elif test_num == 9:
    print(nx.is_connected(graph))
elif test_num == 10:  # Test one shot DP method with dfs tree, compare to tests 2 and 4.
    t_start = time.time()
    for num_districts in range(2, 51):
        initial_assignment = seed_with_one_shot_dfs_tree(num_districts, .05)
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 11:
    t = nx.Graph()
    t.add_edges_from(
        [(0, 2), (2, 1), (0, 4), (4, 3), (0, 6), (6, 5)]
    )
    for v in t:
        t.nodes[v]["pop"] = 1
    t.nodes[0]["pop"] = 0
    partition, num_partitions = sample_uniform_partition_of_tree(
        t,
        num_parts=3,
        min_pop=2,
        max_pop=2,
        pop_col="pop"
    )
    print(num_partitions)
    print(partition)
elif test_num == 12:
    t_start = time.time()
    for num_districts in range(2, 51):
        initial_assignment = seed_with_degree_biased_walk(
            num_districts,
            0.05
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 13:
    t_start = time.time()
    for num_districts in range(2, 51):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=1000
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 14:
    t_start = time.time()
    for num_districts in range(2, 51):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=100000
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 15:
    t_start = time.time()
    for num_districts in range(2, 51):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=100
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 16:
    t_start = time.time()
    for num_districts in range(2, 101):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=500
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 17:
    t_start = time.time()
    for num_districts in range(2, 101):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=10000
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 18:
    graph = nx.grid_graph((10, 10))
    for v in graph:
        graph.nodes[v]["TOTPOP20"] = 1
    total_pop = sum([graph.nodes[n][pop_col] for n in graph.nodes])
    t_start = time.time()
    for num_districts in range(2, 101):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=5,
            plot=5
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t
elif test_num == 19:
    graph = nx.grid_graph((100, 100))
    for v in graph:
        graph.nodes[v]["TOTPOP20"] = 1
    total_pop = sum([graph.nodes[n][pop_col] for n in graph.nodes])
    t_start = time.time()
    for num_districts in range(2, 101):
        initial_assignment = seed_with_cycle_bite_walk(
            num_districts,
            0.05,
            cutoff=100,
            plot=1
        )
        t = time.time()
        print(f"{num_districts} districts: {t - t_start:.1f} seconds")
        t_start = t

