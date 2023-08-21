import scipy.integrate
from scipy.stats import binom
import numpy as np
from fractions import Fraction
from tqdm import tqdm
from scipy.integrate import dblquad
import seaborn as sns
import matplotlib.pyplot as plt

from ballot_generator import ImpartialCulture, PlackettLuce, BradleyTerry
from condorcet import *
from ballot import Ballot
from profile import PreferenceProfile


def read_profile(file):
    ballot_pool = []
    with open(file, "r") as f:
        for line in f:
            cands = line.strip().split(",")
            ballot_pool.append(
                Ballot(ranking=[set(c) for c in cands], weight=Fraction(1))
            )

    return PreferenceProfile(ballots=ballot_pool)


### Theoretical Condorcet Probability from Culture ###

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
    if any([c > 0.55 for c in culture]):
        return 1

    xs = [culture[0] + culture[1] - culture[2] - culture[3] + culture[4] - culture[5],
          culture[0] - culture[1] + culture[2] + culture[3] - culture[4] - culture[5],
          - culture[0] - culture[1] - culture[2] + culture[3] + culture[4] + culture[5]]
    sigmas = [np.sqrt(m * (1 - x ** 2)) for x in xs]
    pis = [x / np.sqrt(1 - x ** 2) for x in xs]

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

    covar_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            covar_matrix[i, j] = m * (zeta_matrix[i, j] - xs[i] * xs[j])

    rho_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            rho_matrix[i, j] = covar_matrix[i, j] / (sigmas[i] * sigmas[j])

    p_A_condo = L(-np.sqrt(m) * pis[0], np.sqrt(m) * pis[2], -rho_matrix[0, 2])
    p_B_condo = L(-np.sqrt(m) * pis[1], np.sqrt(m) * pis[0], -rho_matrix[0, 1])
    p_C_condo = L(-np.sqrt(m) * pis[2], np.sqrt(m) * pis[1], -rho_matrix[1, 2])

    P = p_A_condo + p_B_condo + p_C_condo

    return P


### Applied to Various Ballot Models ###
def pballot_from_interval_PL(ranking, interval):
    pref_interval = interval.copy()
    prob = 1
    for s in ranking:
        if sum(pref_interval.values()) == 0:
            prob *= 1 / np.math.factorial(len(pref_interval))
        else:
            prob *= pref_interval[next(iter(s))] / sum(pref_interval.values())
        del pref_interval[next(iter(s))]
    return prob


def PL_theoretical_condorcet_prob(m, pref_interval):
    rankings = list(it.permutations([set(c) for c in list(pref_interval.keys())]))
    culture = [Ballot(ranking=r, weight=pballot_from_interval_PL(r, pref_interval)) for r in rankings]
    prob = calculate_condorcet_prob(m, culture)
    return prob


def PL_theory_wrapper(a, b):
    return PL_theoretical_condorcet_prob(m=101,
                                         pref_interval={'A': a, 'B': b, 'C': 1 - a - b})


### Empirical Tests ###
def PL_empirical_condorcet_prob(m, pref_interval, n_sims):
    epsilon = 0.00001
    candidates = ['A', 'B', 'C']
    non_zero_pref_interval = {
        c: (p + epsilon) / (1 + 3 * epsilon) for c, p in pref_interval.items()
    }
    # Compute empirical condorcet winner frequency
    count_condorcet_winner = 0
    for _ in range(n_sims):
        pl_election = PlackettLuce(candidates=candidates,
                                   ballot_length=3,
                                   pref_interval_by_bloc={"A": non_zero_pref_interval},
                                   bloc_voter_prop={"A": 1.0})
        pl_ballots = pl_election.generate_profile(number_of_ballots=m)
        if has_condorcet(pl_ballots):
            count_condorcet_winner += 1

    empirical_winner_freq = count_condorcet_winner / n_sims

    return empirical_winner_freq


def PL_empirical_wrapper(a, b):
    return PL_empirical_condorcet_prob(m=101,
                                       pref_interval={'A': a, 'B': b, 'C': 1 - a - b},
                                       n_sims=500)


def triangle_mask(a, b):
    return a + b <= 1


def generate_heatmap(func, mask, granularity, axlim=(0, 1), show=False, outname=None):
    # We want a map of all values on (a, b) in [0, 1]^2 st a + b < 1
    # We choose a granularity of 25
    # That means our array is [0.02, 0.06, 0.10,.., 0.98]^2
    # This is a 25x25 array
    step_size = (axlim[1] - axlim[0]) / granularity
    outputs = np.zeros((granularity, granularity))

    for i in tqdm(range(len(outputs))):
        for j in range(len(outputs[i])):
            a = (i * step_size + 0.5 * step_size) + axlim[0]
            b = (j * step_size + 0.5 * step_size) + axlim[0]
            if mask(a, b):
                outputs[i][j] = func(a, b)
            else:
                outputs[i][j] = np.nan

    np.save(outname + ".npy", outputs)

    plt.figure(figsize=(8, 8))
    sns.heatmap(outputs, vmin=0.75)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.ylim(0, granularity)
    plt.xlim(0, granularity)
    plt.title('Plackett-Luce Condorcet Probabilities')
    if outname is not None: plt.savefig(outname + ".png")
    if show: plt.show()


def alpha_heatmap(alpha, granularity, n_sims, show=False, outname=None):
    outputs = np.zeros((granularity, granularity))
    epsilon = 0.000001

    for _ in range(n_sims):
        pref_interval_vals = np.random.dirichlet([alpha] * 3)
        a, b = pref_interval_vals[0], pref_interval_vals[1]
        if a >= 1:
            a = 1 - epsilon
        if b >= 1:
            b = 1 - epsilon
        i = int(a * granularity)
        j = int(b * granularity)
        outputs[i][j] += 1

    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            outputs[i][j] = outputs[i][j] / n_sims

    if outname: np.save(outname + ".npy", outputs)

    plt.figure(figsize=(8, 8))
    sns.heatmap(outputs)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.ylim(0, granularity)
    plt.xlim(0, granularity)
    plt.title('Distribution of pref intervals for alpha=' + str(alpha))

    if outname is not None: plt.savefig(outname + ".png")
    if show: plt.show()
    plt.close()


def PL_alpha_sequence(dir):
    alphas = [0.05 * i for i in range(1, 160)] + [0.5 * i for i in range(16, 100)]
    alpha_condo_probs = []
    condo_pref_interval_array = np.load(dir + "\\PL_a_b_condorcet.npy")

    # Sometimes we need to fix the theoretical thing
    # for i in range(len(condo_pref_interval_array)):
    #     for j in range(len(condo_pref_interval_array[i])):
    #         if i < 5 or j < 5 or i + j > 95:
    #             condo_pref_interval_array[i][j] = 1
    #         if i + j >= 100:
    #             condo_pref_interval_array[i][j] = 0

    for alpha in tqdm(alphas):
        alpha_str = str(round(alpha, 2)).replace('.', '_')
        outname = "alphas\\dirichlet_pref_interval_alpha_" + alpha_str
        # alpha_heatmap(alpha=alpha,
        #               granularity=100,
        #               n_sims=10 ** 6,
        #               show=False,
        #               outname=outname)

        dirichlet_pref_interval_array = np.load(outname + ".npy")
        alpha_condo_prob = np.sum(np.multiply(condo_pref_interval_array, dirichlet_pref_interval_array))
        alpha_condo_probs.append(alpha_condo_prob)
    np.save(dir + "\\alpha_condo_probs.npy", np.array(alpha_condo_probs))

    alphas = [0.05 * (i + 1) for i in range(160)] + [0.5 * i for i in range(16, 100)]
    alpha_condo_probs = np.array(list(np.load("alpha_condo_probs.npy")))

    ### Make final graph
    plt.plot(alphas, alpha_condo_probs, linestyle='-', marker='', color='r')
    plt.xlim((0, 7.5))
    plt.xlabel('Alpha')
    plt.ylabel('P(Condorcet Winner)')
    plt.title('Emp Strength of Candidate vs Condorcet Probability')
    plt.savefig("PL_alpha_condorcet_emp_medium.png")
    plt.show()


if __name__ == "__main__":
    # generate_heatmap(func=PL_theory_wrapper,
    #                  mask=triangle_mask,
    #                  granularity=100,
    #                  axlim=(0, 1),
    #                  show=True,
    #                  outfile="PL_a_b_theory_heatmap.png")

    generate_heatmap(func=PL_empirical_wrapper,
                     mask=triangle_mask,
                     granularity=2,
                     axlim=(0, 1),
                     show=False,
                     outname="PL_a_b_empirical_heatmap")

    # PL_alpha_sequence("emp_3_cand")
