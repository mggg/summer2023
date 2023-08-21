import pickle
from numpy.random import choice
from collections import defaultdict
import numpy as np
from itertools import permutations, product, combinations
import random

def sum_to_one(list_of_vectors):
    '''
    Fixes small errors in place to make sure vectors sum to 1
    '''
    for v in list_of_vectors:
        n = np.argmax(v) #fix highest value
        v[n] = 1-sum([x for i,x in enumerate(v) if i!=n])

def paired_comparison_mcmc(num_ballots,
                           mean_support_by_race,
                           std_support_by_race,
                           cand_list,
                           vote_portion_by_race,
                           race_list,
                           seeds=None,
                           sample_interval=10,
                           verbose = True):
    #Sample from probability distribution for each race using MCMC - don't explicitly
    #compute probability of each ballot in advance
    #Draw from each race's prob distribution (number of ballots per race dtmd by cvap share)
    ordered_cand_pairs = list(permutations(cand_list,2))
    ballots_list = []

    for race in race_list:
        #make dictionairy of paired comparisons: i.e. prob i>j for all ordered pairs of candidates
        #keys are ordered pair of candidates, values are prob i>j in pair of candidates
        paired_compare_dict = {k: mean_support_by_race[race][k[0]]/(mean_support_by_race[race][k[0]]+mean_support_by_race[race][k[1]]) for k in ordered_cand_pairs}
        #starting ballot for mcmc
        start_ballot = list(np.random.permutation(cand_list))
        #function for evaluating single ballot in MCMC
        #don't need normalization term here! Exact probability of a particular ballot would be
        #output of this fnction divided by normalization term that MCMC allows us to avoid
        track_ballot_prob = []
        def ballot_prob(ballot):
            pairs_list_ballot = list(combinations(ballot,2))
            paired_compare_trunc = {k: paired_compare_dict[k] for k in pairs_list_ballot}
            ballot_prob = np.product(list(paired_compare_trunc.values()))
            return ballot_prob

        #start MCMC with 'start_ballot'
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        race_ballot_list = []
        step = 0
        accept = 0
        while len(race_ballot_list) < num_ballots_race: #range(num_ballots_race):
            #proposed new ballot is a random switch of two elements in ballot before
            proposed_ballot = start_ballot.copy()
            j1,j2 = random.sample(range(len(start_ballot)),2)
            proposed_ballot[j1], proposed_ballot[j2] = proposed_ballot[j2], proposed_ballot[j1]

            #acceptance ratio: (note - symmetric proposal function!)
            accept_ratio = min(ballot_prob(proposed_ballot)/ballot_prob(start_ballot),1)
            #accept or reject proposal
            if random.random() < accept_ratio:
                start_ballot = proposed_ballot
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot)
                accept += 1
            else:
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot)
            step += 1
        ballots_list = ballots_list + race_ballot_list
        if verbose:
            if step > 0:
                print("Acceptance ratio for {} voters: ".format(race), accept/step)
       # plt.plot(track_ballot_prob)
    return ballots_list


def pl_ballots(
    poc_share = 0.33,
    poc_support_for_poc_candidates = 0.7,
    poc_support_for_white_candidates = 0.3,
    white_support_for_white_candidates = 0.8,
    white_support_for_poc_candidates = 0.2,
    num_ballots = 1000,
    num_simulations = 1,
    num_poc_candidates = 2,
    num_white_candidates = 3,
    concentrations = [1.0,1.0,1.0,1.0], #poc_for_poc, poc_for_w, w_for_poc, w_for_w.
    max_ballot_length = None
):
    if max_ballot_length == None:
        max_ballot_length = num_poc_candidates+num_white_candidates
    num_candidates = [num_poc_candidates, num_white_candidates]
    alphas = concentrations
    candidates = ['A'+str(x) for x in range(num_poc_candidates)]+['B'+str(x) for x in range(num_white_candidates)]
    race_of_candidate = {x:x[0] for x in candidates}

    #simulate
    for n in range(num_simulations):
        #get support vectors

        noise0 = list(np.random.dirichlet([alphas[0]]*num_candidates[0]))+list(np.random.dirichlet([alphas[1]]*num_candidates[1]))
        noise1 = list(np.random.dirichlet([alphas[2]]*num_candidates[0]))+list(np.random.dirichlet([alphas[3]]*num_candidates[1]))

        white_support_vector = []
        poc_support_vector = []
        for i, (c, r) in enumerate(race_of_candidate.items()):
            if r == 'A':
                white_support_vector.append((white_support_for_poc_candidates*noise1[i]))
                poc_support_vector.append((poc_support_for_poc_candidates*noise0[i]))
            elif r == 'B':
                white_support_vector.append((white_support_for_white_candidates*noise1[i]))
                poc_support_vector.append((poc_support_for_white_candidates*noise0[i]))

        ballots = []
        numballots = num_ballots
        print('pl', white_support_vector)
        print('pl', poc_support_vector)
        sum_to_one([white_support_vector, poc_support_vector])
        #white
        for i in range(int(numballots*(1-poc_share))):
          ballots.append(
              np.random.choice(list(race_of_candidate.keys()), size=len(race_of_candidate), p=white_support_vector, replace=False)
          )
        #poc
        for i in range(int(numballots*poc_share)):
          ballots.append(
              np.random.choice(list(race_of_candidate.keys()), size=len(race_of_candidate), p=poc_support_vector, replace=False)
          )
        #winners
        ballots = [list(b[:max_ballot_length]) for b in ballots]

    return ballots

def bt_ballots(
    poc_share = 0.33,
    poc_support_for_poc_candidates = 0.7,
    poc_support_for_white_candidates = 0.3,
    white_support_for_white_candidates = 0.8,
    white_support_for_poc_candidates = 0.2,
    num_ballots = 1000,
    num_simulations = 100,
    num_poc_candidates = 2,
    num_white_candidates = 3,
    concentrations = [1.0,1.0,1.0,1.0], #poc_for_poc, poc_for_w, w_for_poc, w_for_w
    max_ballot_length = None
):
    if max_ballot_length == None:
        max_ballot_length = num_poc_candidates+num_white_candidates
    num_candidates = [num_poc_candidates, num_white_candidates]
    alphas = concentrations
    candidates = ['A'+str(x) for x in range(num_poc_candidates)]+['B'+str(x) for x in range(num_white_candidates)]
    race_of_candidate = {x:x[0] for x in candidates}

    #simulate
    for n in range(num_simulations):
        #get support vectors
        noise0 = list(np.random.dirichlet([alphas[0]]*num_candidates[0]))+list(np.random.dirichlet([alphas[1]]*num_candidates[1]))
        noise1 = list(np.random.dirichlet([alphas[2]]*num_candidates[0]))+list(np.random.dirichlet([alphas[3]]*num_candidates[1]))
        white_support_vector = []
        poc_support_vector = []
        for i, (c, r) in enumerate(race_of_candidate.items()):
            if r == 'A':
                white_support_vector.append((white_support_for_poc_candidates*noise1[i]))
                poc_support_vector.append((poc_support_for_poc_candidates*noise0[i]))
            elif r == 'B':
                white_support_vector.append((white_support_for_white_candidates*noise1[i]))
                poc_support_vector.append((poc_support_for_white_candidates*noise0[i]))

        # print(white_support_for_poc_candidates)
        # print(poc_support_for_white_candidates) 
        ballots = []
        numballots = num_ballots
        ballots = paired_comparison_mcmc(
            num_ballots,
            {
                0:{x:poc_support_vector[i] for i,x in enumerate(candidates)},
                1:{x:white_support_vector[i] for i, x in enumerate(candidates)}
            },
            None,
            candidates,
            {0:poc_share, 1:1-poc_share},
            [0,1],
            sample_interval=10,
            verbose=False
        )
        #winners
        ballots = [b[:max_ballot_length] for b in ballots]

    return ballots
