import numpy as np
# from mcmc import gen_transition_probs
from blt_parser import parse
from collections import OrderedDict

def random_walk(mat, num_ballots):
    cands = range(len(mat))
    src = 0
    walk = []
    walk.append(src)
    b_count = 0
    while b_count != num_ballots:
        targ_prob = mat[src]
        # print(targ_prob)
        targ = np.random.choice(a=cands, size=1, p=targ_prob)[0]
        # print(targ)
        walk.append(targ)
        if targ == 0:
            b_count  += 1
        # print(f'step {i}: {src} -> {targ} with p: {targ_prob[targ]}')
        src = targ
    
    # print(walk)
    return walk
    
mat = np.array([[0.,0.4,0.,0.6],
       [0.2,0.,0.8,0.],
       [0.8,0.2,0.,0.],
       [0.,0.66666667,0.33333333,0.]])


def dedup(ballot):
    return list(OrderedDict.fromkeys(ballot))

def truncate(walk, deduplicate=False):

    result = []
    ballot = []
    for num in walk:
        if num != 0:
            ballot.append(num)
        elif ballot:
            result.append(ballot)
            ballot = []

    if ballot:  # In case the last element(s) are not followed by 0
        result.append(ballot)

    if deduplicate:
        result = [dedup(ballot) for ballot in result]

    return ballots_to_profile(result)

def rejection(walk):
    result = []
    ballot = []
    for num in walk:
        if num != 0:
            ballot.append(num)
        elif ballot:
            if len(set(ballot)) == len(ballot):
                result.append(ballot)
            ballot = []

    if ballot:
        if len(set(ballot)) == len(ballot):
                result.append(ballot)  # In case the last element(s) are not followed by 0

    return ballots_to_profile(result)

def dedup(ballot):
    return list(OrderedDict.fromkeys(ballot))

def get_len(ballots):
    return list(map(len, ballots))

def loop_erase(walk):
    result = []
    ballot = []
    for num in walk:
        if num != 0:
            if num in ballot:
                ballot = ballot[:ballot.index(num)] 
            ballot.append(num)
        elif ballot:
            result.append(ballot)
            ballot = []

    if ballot:  # In case the last element(s) are not followed by 0
        result.append(ballot)
    
    return ballots_to_profile(result) 

#election = {(3, 2, 1) : 1, (1, 2): 2, (3, 1, 2): 2}
def ballots_to_profile(ballots):
    profile = {}
    for ballot in ballots:
        if tuple(ballot) not in profile:
            profile[tuple(ballot)] = 0
        profile[tuple(ballot)] += 1

    return profile

def expected_bl(profile):
    n = 0
    y = 0
    for ballot, votes in profile.items():
        y += (len(ballot) * votes)
        n += votes

    return y/n 

def expected_bp(profile):
    n = sum(profile.values())
    bullet = 0
    for ballot, voters in profile.items():
        if len(ballot) == 1:
            bullet += voters

    return bullet/n

# def expected_generated(ballots):
#     y = 0
#     for ballot in ballots:
#         y += len(ballot)
    
#     return y/len(ballots)



if __name__ == '__main__':
    ...
    # election, names, location = parse("Data/edinburgh17-16.blt")
    # mat = gen_transition_probs(election=election)
    # n = 10
    # walk = sample_walk(mat, n)
    # ballots = cut_up_ballots(walk)
    # deduped = list(map(dedup, ballots))
    # le = list(map(loop_erase, ballots))
    # print(get_len(le))
    # print(le)

# ballot = [2, 1, 2, 1, 3]
# print(dedup(ballot))

# print(deduped)
    
