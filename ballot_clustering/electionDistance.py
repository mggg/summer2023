import numpy as np
from numpy import matrix
import itertools
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import ot
import ot.plot
import networkx as nx
from numpy.random import choice
from collections import Counter
from ballots import BallotGen
import time
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from pref_schedule import pref_schedule


# TODO: test and rename functions in compliance with pref_schedule

'''
This class compares lists of preference schedules, and makes an MDS plot 
organized by the source of the ballots in the preference scehdules (e.g. PL or BT-generated)
'''

def earth_mover_dist(electA_distr, electB_distr, cost_matrix):
        #  Solving Earth Mover Distance
        G0 = ot.emd(electA_distr,electB_distr, cost_matrix)
        
        #  Hadamard Product = Earth mover dist between two matrices 
        earth_mover_dist = np.sum(np.multiply(cost_matrix, G0))
        return earth_mover_dist

class electionDistance: 

    # TODO: set generic dist_type that you can invoke 

    # elections_by_type takes list of lists of pref_schedules, organized by type (e.g. PL generated, BT generated) 
    # election_type_names are used to label MDS plot 
    def __init__(self, elections_by_type, election_type_names = [], dist_func = earth_mover_dist):
        self.elections_by_type = elections_by_type
        self.election_type_names = election_type_names 
        
        self.allElections = list(itertools.chain.from_iterable(self.elections_by_type))
        
        self.dist_matrix = [[0 for __ in range(len(self.allElections))] for _ in range(len(self.allElections))]
        for i in range(len(self.allElections)): 
            electionA_vals = [value for key, value in sorted(self.allElections[i].items())]
            electA_distr = np.array(electionA_vals)
            for j in range(i+1,len(self.allElections)):
                electionB_vals = [value for key, value in sorted(self.allElections[j].items())]
                electB_distr = np.array(electionB_vals)

                # TODO: cost_matrix gets called from prefschedule class 
                em_dist = dist_func(electA_distr,electB_distr, pref_schedule.get_cost_matrix())
                self.dist_matrix[i][j] = em_dist
                self.dist_matrix[j][i] = em_dist


    def generate_MDS_plot(self):

        mds = manifold.MDS(
            n_components=2,
            max_iter=3000,
            eps=1e-9,
            dissimilarity="precomputed",
            n_jobs=1,
            normalized_stress="auto",
        )

        pos = mds.fit(np.array(self.dist_matrix)).embedding_
        #print(pos)

        # get indicies for each distribution type set for MDS plotting
        indicies = []
        for i, distribution_set in enumerate(self.elections_by_type):
            if i == 0:
                indicies.append(len(distribution_set))
            else:
                indicies.append(len(distribution_set) + indicies[i - 1])

        point_size = 5
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'black', 'turquoise']

        for n, index in enumerate(indicies):
            if n == 0:
                plt.scatter(pos[0:index, 0], pos[0:index, 1], color= colors[n], lw=0, s=point_size)
            else:
                plt.scatter(pos[indicies[n - 1]:index, 0], pos[indicies[n - 1]:index, 1], color= colors[n], lw=0, s=point_size)


        plt.legend(self.election_type_names)

        # TODO: num_elections * 4? 
       # plt.title(f'Earth Mover Dist: {num_elections} Elections with {len(cand_list)} Candidates')
        plt.show()
                        
    

        


