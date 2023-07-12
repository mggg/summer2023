import matplotlib.pyplot as plt
import pandas as pd
import itertools
from numpy.random import choice
import random
from collections import Counter
import numpy as np

'''
This class generates balllots for PL and BT ballot types 
'''

class BallotGen:

  def __init__(self,num_ballots, cand_list, voter_proportion_by_race, cand_support_interval):
    self.num_ballots = num_ballots
    self.cand_list = cand_list
    self.voter_proportion_by_race = voter_proportion_by_race
    self.cand_support_interval = cand_support_interval
    self.list_of_ballot_distributions = []



  def pl_ballots(self):
      """ Returns a list of PL ballots of length num_ballots, 
      with size of blocks proportional to voter_proportion_by_race """

      ballots_list = []
      for race in self.voter_proportion_by_race.keys():
          num_ballots_race = int(self.num_ballots*self.voter_proportion_by_race[race])##this computes number of voters in this race/block
          cand_support_vec = list(self.cand_support_interval[race].values())##creates the interval of probabilities for candidates supported by this block
          for j in range(num_ballots_race):
              x = tuple(choice(self.cand_list, len(self.cand_list), replace = False, p = cand_support_vec))
              ballots_list.append(x)
      return ballots_list
  
  def bt_ballots(self): 
    """ Returns a list of BT ballots of length num_ballots, 
      with size of blocks proportional to voter_proportion_by_race """

    n=len(self.cand_list)
    k=0
    permutations = list(itertools.permutations(self.cand_list))
    for combo in permutations:##computes (inverse of) the constant of proportionality
      m=1
      for i in range(n):
        for j in range(i+1,n):
          l=0
          for race in self.voter_proportion_by_race.keys():
            l = l+self.voter_proportion_by_race[race]*(self.cand_support_interval[race][combo[i]]/(self.cand_support_interval[race][combo[i]]+self.cand_support_interval[race][combo[j]]))
          if j!=i:
            m=m*l
      k=k+m

    weights = []

    for combo in permutations:
      prob=1
      for i in range(n):
        for j in range(i+1,n):
          l=0
          for race in self.voter_proportion_by_race.keys():
            l = l+self.voter_proportion_by_race[race]*(self.cand_support_interval[race][combo[i]]/( self.cand_support_interval[race][combo[i]]+self.cand_support_interval[race][combo[j]]))
          prob=prob*l
      weights.append(prob/k)##we're giving each permutation of cand_list a weight

    x = choice(range(len(permutations)), self.num_ballots, replace=True, p = weights)
    return list(permutations[i] for i in x)

    
  
  
  def pl_ballot_count(self,num_elections, normalize = False):
    """ Returns a list of dictionaries of length num_elections, where each dictionary gives the count of each ballot type
     for that election. If normalize == true, divides all counts by number of ballots in election (so gives the 
     distribution of ballot types)"""

    # create list of lists of all possible ballots, then default dictionary where count of each ballot is 0
    # note that we exclude ballots where num_cands - 1 candidates are ranked 
    ballot_perms = [list(itertools.permutations(self.cand_list, num_ranked)) 
                         for num_ranked in range(1, len(self.cand_list) + 1) if num_ranked != len(self.cand_list) - 1]

    # consolidates ballot_perms into a single list 
    cand_p_list = list(itertools.chain.from_iterable(ballot_perms))

    cand_p_dict ={}
    for i in cand_p_list:
        cand_p_dict[i] = 0

    elections_pl = []
    for i in range(num_elections):
        pl_ballots = self.pl_ballots()

        # create a dictionary-like object that stores the count of each ballot type
        pl_perf_count = Counter(pl_ballots)

        # takes the union of this with the empty dictionary, giving priority to the counts in the latter 
        pl_perf_count = cand_p_dict | pl_perf_count
        
        # Normalize 
        if normalize == True:
          for key in pl_perf_count.keys():
            pl_perf_count[key] = pl_perf_count[key] / self.num_ballots
        elections_pl.append(pl_perf_count)
    
    self.list_of_ballot_distributions = elections_pl
    return elections_pl
  
  


  def bt_ballot_count(self,num_elections,normalize = False):   
    """ Returns a list of dictionaries of length num_elections, where each dictionary gives the count of each ballot type
     for that election. If normalize == true, divides all values by number of ballots in election"""
    # TODO: consolidate with pf_ballot_count (they basically do the same thing)


    # create list of lists of all possible ballots, then default dictionary where count of each ballot is 0
    
    # note that we exclude ballots where num_cands - 1 candidates are ranked 
    ballot_perms = [list(itertools.permutations(self.cand_list, num_ranked)) 
                         for num_ranked in range(1, len(self.cand_list) + 1) if num_ranked != len(self.cand_list) - 1]

    # consolidates ballot_perms into a single list 
    cand_p_list = list(itertools.chain.from_iterable(ballot_perms))

    cand_p_dict ={}
    for i in cand_p_list:
        cand_p_dict[i] = 0
    elections_bt = []
    diff =[]
    for i in range(num_elections):
        bt_ballots = self.bt_ballots()
        bt_perf_count = Counter(bt_ballots)
        bt_keys = bt_perf_count.keys()
        bt_perf_count = cand_p_dict | bt_perf_count

        # Normalize 
        if normalize == True:
          for key in bt_perf_count.keys():
            bt_perf_count[key] = bt_perf_count[key] / self.num_ballots

        elections_bt.append(bt_perf_count)

    self.list_of_ballot_distributions = elections_bt
    return elections_bt
