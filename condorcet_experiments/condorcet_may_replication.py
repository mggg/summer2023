import scipy.integrate
from scipy.stats import binom
import numpy as np
from fractions import Fraction
from tqdm import tqdm
from scipy.integrate import dblquad
import seaborn as sns
import matplotlib.pyplot as plt

from ballot_generator import IC, PlackettLuce, BradleyTerry
from condorcet import *
from ballot import Ballot
from profile import PreferenceProfile

def L(h, k, rho):
    return dblquad(g, h, np.inf, k, np.inf, args=(rho,))[0]


def g(x, y, rho):
    coeff = 1 / (2 * np.pi * np.sqrt(1 - rho ** 2))
    exponential = np.exp(-0.5 * (x ** 2 - 2 * rho * x * y + y ** 2) / (1 - rho ** 2))
    return coeff * exponential


def calculate_condorcet_prob(m, ballot_culture):
    epsilon = 0.00001
    culture = [ballot.weight for ballot in ballot_culture]
    culture = [(c + epsilon) / (sum(culture) + 6 * epsilon) for c in culture]
    #print(culture)

    xs = [culture[0] + culture[1] - culture[2] - culture[3] + culture[4] - culture[5],
          culture[0] - culture[1] + culture[2] + culture[3] - culture[4] - culture[5],
          - culture[0] - culture[1] - culture[2] + culture[3] + culture[4] + culture[5]]
    #print(xs)
    sigmas = [np.sqrt(m * (1 - x ** 2)) for x in xs]
    #print(sigmas)
    pis = [x / np.sqrt(1 - x ** 2) for x in xs]
    #print(pis)

    zeta_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                zeta_matrix[i, j] = 1
            elif (i, j) == (0, 1) or (i, j) == (1, 0):
                zeta_matrix[i, j] = culture[0] - culture[1] - culture[2] - \
                                    culture[3] - culture[4] + culture[5]
            elif (i, j) == (0, 2) or (i, j) == (2, 0):
                zeta_matrix[i, j] = - culture[0] - culture[1] + culture[2] - \
                                    culture[3] + culture[4] - culture[5]
            elif (i, j) == (1, 2) or (i, j) == (2, 1):
                zeta_matrix[i, j] = - culture[0] + culture[1] - culture[2] + \
                                    culture[3] - culture[4] - culture[5]
    #print(zeta_matrix)

    covar_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            covar_matrix[i, j] = m * (zeta_matrix[i, j] - xs[i] * xs[j])
    #print(covar_matrix)

    rho_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            rho_matrix[i, j] = covar_matrix[i, j] / (sigmas[i] * sigmas[j])
    #print(rho_matrix)

    p_A_condo = L(-np.sqrt(m) * pis[0], np.sqrt(m) * pis[2], -rho_matrix[0, 2])
    p_B_condo = L(-np.sqrt(m) * pis[1], np.sqrt(m) * pis[0], -rho_matrix[0, 1])
    p_C_condo = L(-np.sqrt(m) * pis[2], np.sqrt(m) * pis[1], -rho_matrix[1, 2])
    #print(p_A_condo, p_B_condo, p_C_condo)

    P = p_A_condo + p_B_condo + p_C_condo

    return P


def pballot_from_interval_PL(ranking, interval):
    pref_interval = interval.copy()
    prob = 1
    for c in ranking:
        prob *= pref_interval[next(iter(c))] / sum(pref_interval.values())
        del pref_interval[next(iter(c))]
    return prob


def PL_theoretical_condorcet_prob(m, pref_interval):
    rankings = list(it.permutations([set(c) for c in list(pref_interval.keys())]))
    culture = [Ballot(ranking=r, weight=pballot_from_interval_PL(r, pref_interval)) for r in rankings]
    prob = calculate_condorcet_prob(m, culture)
    return prob


if __name__ == "__main__":
    culture = [Ballot(ranking=[{'A'}, {'B'}, {'C'}], weight=1 / 3),
               Ballot(ranking=[{'A'}, {'C'}, {'B'}], weight=0),
               Ballot(ranking=[{'B'}, {'A'}, {'C'}], weight=0),
               Ballot(ranking=[{'B'}, {'C'}, {'A'}], weight=1 / 3),
               Ballot(ranking=[{'C'}, {'A'}, {'B'}], weight=1 / 3),
               Ballot(ranking=[{'C'}, {'B'}, {'A'}], weight=0)]
    print([round(ballot.weight, 3) for ballot in culture])
    print(1 - round(calculate_condorcet_prob(m=101, ballot_culture=culture), 4))

    culture = [Ballot(ranking=[{'A'}, {'B'}, {'C'}], weight=10/33),
               Ballot(ranking=[{'A'}, {'C'}, {'B'}], weight=1/33),
               Ballot(ranking=[{'B'}, {'A'}, {'C'}], weight=10/33),
               Ballot(ranking=[{'B'}, {'C'}, {'A'}], weight=1/33),
               Ballot(ranking=[{'C'}, {'A'}, {'B'}], weight=10/33),
               Ballot(ranking=[{'C'}, {'B'}, {'A'}], weight=1/33)]
    print([round(ballot.weight, 3) for ballot in culture])
    print(1 - round(calculate_condorcet_prob(m=101, ballot_culture=culture), 4))

    culture = [Ballot(ranking=[{'A'}, {'B'}, {'C'}], weight=1 / 4),
               Ballot(ranking=[{'A'}, {'C'}, {'B'}], weight=1 / 8),
               Ballot(ranking=[{'B'}, {'A'}, {'C'}], weight=1 / 8),
               Ballot(ranking=[{'B'}, {'C'}, {'A'}], weight=1 / 4),
               Ballot(ranking=[{'C'}, {'A'}, {'B'}], weight=1 / 8),
               Ballot(ranking=[{'C'}, {'B'}, {'A'}], weight=1 / 8)]
    print([round(ballot.weight, 3) for ballot in culture])
    print(1 - round(calculate_condorcet_prob(m=101, ballot_culture=culture), 4))

    culture = [Ballot(ranking=[{'A'}, {'B'}, {'C'}], weight=20 / 63),
               Ballot(ranking=[{'A'}, {'C'}, {'B'}], weight=1 / 63),
               Ballot(ranking=[{'B'}, {'A'}, {'C'}], weight=1 / 63),
               Ballot(ranking=[{'B'}, {'C'}, {'A'}], weight=20 / 63),
               Ballot(ranking=[{'C'}, {'A'}, {'B'}], weight=20 / 63),
               Ballot(ranking=[{'C'}, {'B'}, {'A'}], weight=1 / 63)]
    print([round(ballot.weight, 3) for ballot in culture])
    print(1 - round(calculate_condorcet_prob(m=101, ballot_culture=culture), 4))

    culture = [Ballot(ranking=[{'A'}, {'B'}, {'C'}], weight=1 / 6),
               Ballot(ranking=[{'A'}, {'C'}, {'B'}], weight=1 / 6),
               Ballot(ranking=[{'B'}, {'A'}, {'C'}], weight=1 / 6),
               Ballot(ranking=[{'B'}, {'C'}, {'A'}], weight=1 / 6),
               Ballot(ranking=[{'C'}, {'A'}, {'B'}], weight=1 / 6),
               Ballot(ranking=[{'C'}, {'B'}, {'A'}], weight=1 / 6)]
    print([round(ballot.weight, 3) for ballot in culture])
    print(1 - round(calculate_condorcet_prob(m=101, ballot_culture=culture), 4))


