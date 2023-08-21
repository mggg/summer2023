import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import math
from fractions import Fraction

from ballot import Ballot
from profile import PreferenceProfile


# Returns 1 if cand1 > cand2, -1 if cand2 > cand1, 0 if cand1 ~ cand2
def higher_rank(cand1, cand2, ballot: Ballot):
    for s in ballot.ranking:
        if (cand1 in s) and (cand2 in s):
            return 0
        elif cand1 in s:
            return 1
        elif cand2 in s:
            return -1
    return 0


def head_2_head(cand1, cand2, profile: PreferenceProfile):
    total = 0
    for ballot in profile.get_ballots():
        total += ballot.weight * higher_rank(cand1, cand2, ballot)
    return total


def condorcet_matrix(profile: PreferenceProfile):
    candidates = profile.get_candidates()
    cand_matrix = np.zeros((len(candidates), len(candidates)))
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i != j:
                head2head = head_2_head(candidates[i], candidates[j], profile)
                cand_matrix[i][j] = head2head
    return cand_matrix


def has_condorcet(profile: PreferenceProfile):
    cmatrix = condorcet_matrix(profile)
    for c, row in zip(['A', 'B', 'C'], cmatrix):
        if all(element >= 0 for element in row) and (row == 0).sum() == 1:
            return True
    return False


def h2h_graph(profile: PreferenceProfile):
    return None


def main():
    bl_list = [
        Ballot(id=None, ranking=[{'W1'}], weight=Fraction(10, 1), voters=None),
        Ballot(id=None, ranking=[{'W2'}, {'W1'}, {'C1'}, {'C2'}], weight=Fraction(10, 1), voters=None),
        Ballot(id=None, ranking=[{'W2'}, {'W1'}], weight=Fraction(10, 1), voters=None)
    ]
    toy_pp = PreferenceProfile(ballots=bl_list)
    num_elected = len(toy_pp.get_candidates())
    test_ballotFill = ballot_fill(profile=toy_pp, num_ranking=num_elected)


if __name__ == "__main__":
    pass
