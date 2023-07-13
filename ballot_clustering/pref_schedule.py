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
import EM_dists 

##the packages below are all from generalRCV
from compute_winners import rcv_run
from distinctipy import get_colors
from vote_transfers import cincinnati_transfer

Graphs = {}


class pref_schedule:
    def __init__(self,ballot_dict):
        cand_list = []
        for key in ballot_dict.keys():
            for i in key:
                cand_list.append(i)
                
        self.cands = list(set(cand_list)) 
        self.num_cands = len(self.cands)
        
        if self.num_cands not in Graphs.keys():
            pref_schedule.build_graph(self.num_cands)
        
        
        all_ballots = Graphs[self.num_cands].nodes
        di = {}
        for ballot in all_ballots:
            di[ballot] = 0
        
        self.ballot_dict = di | ballot_dict
        
        
        self.clean()
        
        self.num_voters = sum(ballot_dict.values())
        
        
    def clean(self): #deletes empty ballots, changes n-1 length ballots to n length ballots and updates counts
        di = self.ballot_dict.copy()
        
        for ballot in di.keys():
            if len(ballot)==0:
                self.ballot_dict.pop(ballot)
            elif len(ballot)==self.num_cands-1:
                for i in self.cands:
                    if i not in ballot:
                        self.ballot_dict[ballot+tuple([i])]+= di[ballot]    
                        self.ballot_dict.pop(ballot)
                        break
                
        
    def visualize(self, neighborhoods = {}):
        Gc = Graphs[self.num_cands]    
        if neighborhoods == {}:
            
            
            self.clean()
            NONE = (1,1,1)
            cols = get_colors(self.num_cands, [NONE])
            node_cols = []
            ballots = list(Gc.nodes)
        
            for bal in Gc.nodes:
                if self.ballot_dict[bal]!=0:
                    i = self.cands.index(bal[0])
                    node_cols.append(cols[i])
                else:
                    node_cols.append(NONE)
            ##want to include number of votes as part of labels,  color ballots with 0 votes grey
        
            nx.draw_networkx(Gc,with_labels = True, node_color = node_cols)
            
        else:
            NONE = (1,1,1)
            cols = get_colors(len(neighborhoods), [NONE])
            node_cols = []
            centers = list(neighborhoods.keys())
            
            for bal in Gc.nodes:
                found = False
                for i in range(len(centers)):    
                    if bal in (neighborhoods[centers[i]])[0].nodes:
                        node_cols.append(cols[i])
                        found = True
                        break ##breaks the inner for loop
                if not found:
                    node_cols.append(NONE)
            nx.draw_networkx(Gc, with_labels = True, node_color = node_cols)
        
        return
           
        
    @staticmethod
    def show_all_ballot_types(n):
        if n not in Graphs.keys():
            pref_schedule.build_graph(n)
        Gc = Graphs[n]
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
            
            if n-1 not in Graphs.keys(): 
            # make the adjacency graph of size (n - 1)
                pref_schedule.build_graph(n-1)
            G_prev = Graphs[n-1]
            for i in range(1,n+1):
                # add the node for the bullet vote i
                Gc.add_node(tuple([i]))

                # make the subgraph for the ballots where i is ranked first
                G_corner = pref_schedule.relabel(G_prev,i,n)
      
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

        Graphs[n] = Gc
        return

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

        
    def rcv_results(self, num_seats, transfer_method, verbose_bool = False): 
        ballot_list = []
        for ballot in self.ballot_dict.keys():
            for i in range(self.ballot_dict[ballot]):
                ballot_list.append(list(ballot))
        
        l= rcv_run(ballot_list, self.cands, num_seats, transfer_method, verbose_bool)
        return tuple(l)
    
    
    
    
    def compare(self, new_pref, dist_type):
        return  ##to be completed
    
    def compare_rcv_results(self, new_pref):
        return ##to be completed
    
    def subgraph_neighborhood(self,center,radius = 2):
        return nx.ego_graph(Graphs[self.num_cands],center,radius)
    
    def k_heaviest_neighborhoods(self, k=2, radius=2):
        cast_ballots = set([x for x in self.ballot_dict.keys() if self.ballot_dict[x] > 0]) ##has 
            
        max_balls = {}
            
        for i in range(k):
            weight = 0
            if len(cast_ballots)==0:
                break
            for center in cast_ballots:
                tmp = 0
                ball = self.subgraph_neighborhood(center, radius)
                relevant = cast_ballots.intersection(set(ball.nodes))##cast ballots inside the ball
                for node in relevant: 
                    tmp += self.ballot_dict[node]

                if tmp>weight:
                    weight = tmp
                    max_center = center
                    max_ball = ball 
                
            not_cast_in_max_ball = set(max_ball.nodes).difference(cast_ballots)
            max_ball.remove_nodes_from(not_cast_in_max_ball)
            max_balls[max_center] = (max_ball, weight)
                
            cast_ballots =  cast_ballots.difference(set(max_ball.nodes))
                
        return max_balls

   

