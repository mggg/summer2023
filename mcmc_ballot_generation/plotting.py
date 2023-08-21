from blt_parser import parse
from mcmc import gen_transition_probs
from randomwalk import expected_bl, expected_bp, truncate, rejection, loop_erase, random_walk

from clustering import *
from tqdm import tqdm
from intervals import *
import numpy as np 
import matplotlib.pyplot as plt
import os


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
        axs[i].bar(X_axis - 0.2, intervals[i1], bar_width)
        axs[i].bar(X_axis + 0.2, intervals[i2], bar_width)
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
        axs[i].bar(X_axis - 0.2, intervals[i1], bar_width)
        axs[i].bar(X_axis + 0.2, intervals[i2], bar_width)
        axs[i].text(0, max(intervals[0] + intervals[1])/2, str(-occurrences), ha="right")
        if extra:
            return


def viz_expected_lengths(directory, prob_func, bullet):
    elections = os.listdir(directory)
    names = []
    gen = []
    actual = []
    dedup = []
    loops = []
    rejects = []
    y_title = 'Percent Bullet Vote'

    for idx, election_path in enumerate(elections):
        election, cands, _ = parse(f'{directory}/{election_path}')
        matrix = gen_transition_probs(election, count_completes=True)
        walk = random_walk(matrix, 1000)

        ballots = truncate(walk)
        dedup_ballots = truncate(walk, deduplicate=True)
        le_ballots = loop_erase(walk)
        reject_ballots = rejection(walk)

        gen.append(prob_func(ballots))
        actual.append(prob_func(election))
        dedup.append(prob_func(dedup_ballots))
        loops.append(prob_func(le_ballots))
        rejects.append(prob_func(reject_ballots))
        names.append(f'Election-{idx} ({len(cands)} candidates)') 

    x = np.arange(len(names))
    bar_width = 0.1

    plt.figure(figsize=(10, 5))
    plt.bar(x-bar_width, gen, width=bar_width, label='Generated')
    plt.bar(x, actual, width=bar_width, label='Actual')
    plt.bar(x+bar_width, dedup, width=bar_width, label='Deduplicated')
    plt.bar(x+bar_width*2, loops, width=bar_width, label='Loop erase')
    plt.bar(x+bar_width*3, rejects, width=bar_width, label='Rejection')

    plt.xlabel('Election')
    plt.ylabel(y_title)
    plt.xticks(x)
    plt.xticks(rotation=-90)

    return
    



        