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
#import EM_dists 
from compute_winners import rcv_run
from distinctipy import get_colors
from vote_transfers import cincinnati_transfer

class prefSchedule:
        
    def __init__(self,num_cands,ballot_dict):
        self.ballot_dict = ballot_dict
        cand_list = []
        for key in self.ballot_dict.keys():
            for i in key:
                cand_list.append(i)
                
        self.cands = list(set(cand_list))
         
        self.num_cands = num_cands
        self.num_voters = sum(ballot_dict.values())
        
    def visualize(self, Gc):
        NONE = (1,1,1)
        cols = get_colors(self.num_cands, [NONE])
        node_cols = []
        ballots = list(Gc.nodes)
        
        for bal in Gc.nodes:
            if bal in self.ballot_dict.keys():
                i = self.cands.index(bal[0])
                node_cols.append(cols[i])
            else:
                node_cols.append(NONE)
        ##want to include number of votes as part of labels,  color ballots with 0 votes grey
        
        nx.draw_networkx(Gc,with_labels = True, node_color = node_cols)
        plt.show()
        return
           
        
    @staticmethod
    def show_all_ballot_types(n):
        Gc = pref_schedule.build_graph(n)
        nx.draw(Gc, with_labels = True)
        plt.show()
        

        
    @staticmethod
    def build_graph(n):
        
        """ Makes the adjacency graph for complete and incomplete ballots
        with n candidates """

        Gc = nx.Graph()

        # base cases
        if n==1:
            Gc.add_nodes_from([(1)])
        elif n==2:
            Gc.add_nodes_from([(1,2),(2,1)])
            Gc.add_edges_from([((1,2), (2,1))])

        else:
            # make the adjacency graph of size (n - 1)
            G_prev = prefSchedule.build_graph(n-1)
           
            for i in range(1,n+1):
                # add the node for the bullet vote i
                Gc.add_node(tuple([i]))

                # make the subgraph for the ballots where i is ranked first
                G_corner = prefSchedule.relabel(G_prev,i,n)
      
                # add the components from that graph to the larger graph
                Gc.add_nodes_from(G_corner.nodes)
                Gc.add_edges_from(G_corner.edges)

                # connect the bullet vote node to the appropriate verticies
                if n == 3:
                    Gc.add_edges_from([(k,tuple([i])) for k in G_corner.nodes])
                else:
                    Gc.add_edges_from([(k,tuple([i])) for k in G_corner.nodes if len(k) == 2])
            
            nodes = Gc.nodes

            # add the additional edges corresponding to swapping the order of the
            # first two candidates
            new_edges = []
            for k in nodes:
                if len(k)==2:
                    new_edges.append(((k[0],k[1]),(k[1],k[0])))
                elif len(k)>2:
                    l = list(k)
                    a = l[0]
                    b=l[1]
                    new_edges.append((tuple([a]+[b]+l[2:]), tuple([b]+[a]+l[2:])))

            Gc.add_edges_from(new_edges)

        return Gc

    @staticmethod
    def relabel(gr, new_label, num_cands):
        """ Takes the graph with n - 1 candidates and relabels it to become the
     subgraph in the n-candidate graph of ballots that have 'new label'
    ranked first """

        node_map = {}
        graph_nodes = list(gr.nodes)


        for k in graph_nodes:
            # add the value of new_label to every entry in every ballot
            tmp = [new_label+y for y in k]

            # reduce everything mod new_label
            for i in range(len(tmp)):
                if tmp[i]> num_cands:
                    tmp[i] = tmp[i]- num_cands
            node_map[k] = tuple([new_label]+tmp)

        return nx.relabel_nodes(gr, node_map)

    @staticmethod
    def generate_cost_matrix(self):
        """
        Generates a cost matrix representing the shortest distance between each pair of nodes
        in the ballot graph. Used to calculate earth-mover distances
        """
        graph = prefSchedule.build_graph(self.num_cands)
        # Floyd Warshall Shortest Distance alorithm. Returns a dictionary of shortest path for each node
        FW_dist_dict = nx.floyd_warshall(graph)
        keysList = list(FW_dist_dict.keys())
        keysList.sort()
        cost_matrix = np.zeros((len(keysList), len(keysList)))
        for i in range(len(keysList)):
            node_dict = FW_dist_dict[keysList[i]]
            cost_col = [value for key, value in sorted(node_dict.items())]
            cost_matrix[i] = cost_col
        return cost_matrix
    
    @staticmethod
    def generate_interval_cost_matrix(self, pref_interval):
        """
        Generates a cost matrix representing the shortest distance between each pair of nodes
        in the ballot graph. Used to calculate earth-mover distances
        """
        graph = prefSchedule.build_graph(self.num_cands)
        # Floyd Warshall Shortest Distance alorithm. Returns a dictionary of shortest path for each node
        FW_dist_dict = nx.floyd_warshall(graph)
        keysList = list(FW_dist_dict.keys())
        keysList.sort()
        cost_matrix = np.zeros((len(keysList), len(keysList)))
        for i in range(len(keysList)):
            node_dict = FW_dist_dict[keysList[i]]
            cost_col = [value for key, value in sorted(node_dict.items())]
            cost_matrix[i] = cost_col
        return cost_matrix
    

        
    def rcv_results(self, num_seats, transfer_method, verbose_bool = False): 
        ballot_list = []
        for key in self.ballot_dict.keys():
            for i in range(self.ballot_dict[key]):
                ballot_list.append(list(key))
        
        l= rcv_run(ballot_list, self.cands, num_seats, transfer_method, verbose_bool)
        return tuple(l)
    
    
    
    def compare(self, new_pref, dist_type):
        ##return xxxxx.dist_type(self, new_pref)
        return
    
    def compare_rcv_results(self, new_pref):
        return bubble_sort_dist(self.rcv_results, new_pref.rcv_results)

