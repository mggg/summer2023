import numpy as np


def round_dict(data_dict, d):
    return {k: round(v, d) for k, v in data_dict.items()}


def rej_le_ballot_probs(corpus_probs):
    a, b, c, d = corpus_probs['A'], corpus_probs['B'], corpus_probs['AB'], corpus_probs['BA']

    n = 1 - c * d / ((1 - a) * (1 - b))
    p_A = (1 / n) * (a + c) * (a + d) / (1 - b)
    p_B = (1 / n) * (b + d) * (b + c) / (1 - a)
    p_AB = (1 / n) * (a + c) * (c / (1 - b)) * (b + c) / (1 - a)
    p_BA = (1 / n) * (b + d) * (d / (1 - a)) * (a + d) / (1 - b)

    float_norm = p_A + p_B + p_AB + p_BA
    generated_ballot_probs = {'A': p_A / float_norm, 'B': p_B / float_norm,
                              'AB': p_AB / float_norm, 'BA': p_BA / float_norm}
    return generated_ballot_probs


def dedup_lt_ballot_probs(corpus_probs):
    a, b, c, d = corpus_probs['A'], corpus_probs['B'], corpus_probs['AB'], corpus_probs['BA']

    p_A = (a + c) * (a + d) / (1 - b)
    p_B = (b + d) * (b + c) / (1 - a)
    p_AB = (a + c) * (c / (1 - b))
    p_BA = (b + d) * (d / (1 - a))

    float_norm = p_A + p_B + p_AB + p_BA
    generated_ballot_probs = {'A': p_A / float_norm, 'B': p_B / float_norm,
                              'AB': p_AB / float_norm, 'BA': p_BA / float_norm}
    return generated_ballot_probs


def stochastic_dedup_ballot_probs(corpus_probs):
    a, b, c, d = corpus_probs['A'], corpus_probs['B'], corpus_probs['AB'], corpus_probs['BA']

    n = c * d / ((1 - a) * (1 - b))
    p_A = (a + c) * (a + d) / (1 - b)
    p_B = (b + d) * (b + c) / (1 - a)
    p_AB = (a + c) * (c / (1 - b)) * (b + c) / (1 - a) + n * c / (c + d)
    p_BA = (b + d) * (d / (1 - a)) * (a + d) / (1 - b) + n * d / (c + d)

    float_norm = p_A + p_B + p_AB + p_BA
    generated_ballot_probs = {'A': p_A / float_norm, 'B': p_B / float_norm,
                              'AB': p_AB / float_norm, 'BA': p_BA / float_norm}
    return generated_ballot_probs


def distance_in_prob_profile(corpus_probs, generated_ballot_probs):
    p_distances = [(corpus_probs[b] - generated_ballot_probs[b]) ** 2 for b in BALLOTS]
    return np.sqrt(sum(p_distances))


def find_worst_match_2cand_mc_randomized():
    rej_le_param_dists = {}
    dedup_lt_param_dists = {}
    for _ in range(100000):
        corpus_nums = np.random.dirichlet([1] * 4)
        corpus_probs = {b: c for b, c in zip(BALLOTS, corpus_nums)}

        rej_le_gen_ballot_probs = rej_le_ballot_probs(corpus_probs)
        dedup_lt_gen_ballot_probs = dedup_lt_ballot_probs(corpus_probs)

        rej_le_score = distance_in_prob_profile(corpus_probs, rej_le_gen_ballot_probs)
        dedup_lt_score = distance_in_prob_profile(corpus_probs, dedup_lt_gen_ballot_probs)

        rej_le_param_dists[tuple(corpus_nums)] = rej_le_score
        dedup_lt_param_dists[tuple(corpus_nums)] = dedup_lt_score

    print("Rejection/Loop Erasure")
    percentile_output_dict(rej_le_param_dists)
    print("\nDeduplication/Loop Termination")
    percentile_output_dict(dedup_lt_param_dists)


def mc_match_test_corpi_of_interest():
    eps = 0.00001
    corpus_num_list = [[1 - 3 * eps, eps, eps, eps],
                        [eps, eps, 1 - 3 * eps, eps],
                        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                        [(1 - eps) / 3, (1 - eps) / 3, (1 - eps) / 3, eps],
                        [(1 - eps) / 2, (1 - eps) / 6, (1 - eps) / 3, eps],
                        [eps, (1 - eps) / 3, (1 - eps) / 3, (1 - eps) / 3],
                        [eps, (1 - eps) / 3, (1 - eps) / 6, (1 - eps) / 2],
                        [(1 - eps) / 2, (1 - eps) / 2, eps, eps],
                        [1 / 3, 1 / 3, 1 / 6, 1 / 6],
                        [eps, eps, (1 - eps) / 2, (1 - eps) / 2],
                        [eps, eps, (1 - eps) / 4, 3 * (1 - eps) / 4],
                        [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                        [(1 - eps) / 2, eps, eps, (1 - eps) / 2],
                        [1 / 3, 1 / 6, 1 / 6, 1 / 3]]

    for corpus_nums in corpus_num_list:
        corpus_probs = {b: c for b, c in zip(BALLOTS, corpus_nums)}
        rej_le_gen_ballot_probs = rej_le_ballot_probs(corpus_probs)
        dedup_lt_gen_ballot_probs = dedup_lt_ballot_probs(corpus_probs)
        stoc_dedup_gen_ballot_probs = stochastic_dedup_ballot_probs(corpus_probs)

        rej_le_score = distance_in_prob_profile(corpus_probs, rej_le_gen_ballot_probs)
        dedup_lt_score = distance_in_prob_profile(corpus_probs, dedup_lt_gen_ballot_probs)
        stoc_dedup_score = distance_in_prob_profile(corpus_probs, stoc_dedup_gen_ballot_probs)

        print("Corpus Probs:", round_dict(corpus_probs, 3))
        print("Rejection/Loop Erasure Ballot Probs:", round_dict(rej_le_gen_ballot_probs, 3))
        print("Rejection/Loop Erasure Score:", rej_le_score)
        print("Deduplication/Loop Termination Ballot Probs:", round_dict(dedup_lt_gen_ballot_probs, 3))
        print("Deduplication/Loop Termination Score:", dedup_lt_score)
        print("Stochastic Deduplication Ballot Probs:", round_dict(stoc_dedup_gen_ballot_probs, 3))
        print("Stochastic Deduplication Score:", stoc_dedup_score)
        print("\n")


def percentile_output_dict(data_dict):
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1])
    values = np.array([item[1] for item in sorted_items])  # Extract the values as a numpy array
    percentiles = np.arange(0, 101, 10)  # Generate the percentiles from 0 to 100 with a step of 10

    for p in percentiles:
        value_at_percentile = np.percentile(values, p)
        idx = np.searchsorted(values, value_at_percentile)  # Find the index corresponding to the percentile
        key_at_percentile = sorted_items[idx][0]
        print(p, key_at_percentile, value_at_percentile)


def iterate_mc_corpi_of_interest():
    eps = 0.00001
    corpus_num_list = [[1 - 3 * eps, eps, eps, eps],
                       [eps, eps, 1 - 3 * eps, eps],
                       [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                       [(1 - eps) / 3, (1 - eps) / 3, (1 - eps) / 3, eps],
                       [(1 - eps) / 2, (1 - eps) / 6, (1 - eps) / 3, eps],
                       [eps, (1 - eps) / 3, (1 - eps) / 3, (1 - eps) / 3],
                       [eps, (1 - eps) / 3, (1 - eps) / 6, (1 - eps) / 2],
                       [(1 - eps) / 2, (1 - eps) / 2, eps, eps],
                       [1 / 3, 1 / 3, 1 / 6, 1 / 6],
                       [eps, eps, (1 - eps) / 2, (1 - eps) / 2],
                       [eps, eps, (1 - eps) / 4, 3 * (1 - eps) / 4],
                       [1 / 6, 1 / 6, 1 / 3, 1 / 3],
                       [(1 - eps) / 2, eps, eps, (1 - eps) / 2],
                       [1 / 3, 1 / 6, 1 / 6, 1 / 3]]
    cap = 100000

    for corpus_nums in corpus_num_list:
        corpus_probs = {b: c for b, c in zip(BALLOTS, corpus_nums)}
        iters = 0
        rej_le_probs = corpus_probs
        while distance_in_prob_profile(rej_le_probs, rej_le_ballot_probs(rej_le_probs)) > 0.00000001 and iters < cap:
            rej_le_probs = rej_le_ballot_probs(rej_le_probs)
            iters += 1

        iters = 0
        dedup_lt_probs = corpus_probs
        while distance_in_prob_profile(dedup_lt_probs,
                                       dedup_lt_ballot_probs(dedup_lt_probs)) > 0.00000001 and iters < cap:
            dedup_lt_probs = dedup_lt_ballot_probs(dedup_lt_probs)
            iters += 1

        iters = 0
        dedup_lt_probs = corpus_probs
        while distance_in_prob_profile(dedup_lt_probs,
                                       dedup_lt_ballot_probs(dedup_lt_probs)) > 0.00000001 and iters < cap:
            dedup_lt_probs = dedup_lt_ballot_probs(dedup_lt_probs)
            iters += 1

        iters = 0
        dedup_lt_probs = corpus_probs
        while distance_in_prob_profile(dedup_lt_probs,
                                       dedup_lt_ballot_probs(dedup_lt_probs)) > 0.00000001 and iters < cap:
            dedup_lt_probs = dedup_lt_ballot_probs(dedup_lt_probs)
            iters += 1

        iters = 0
        stoc_dedup_probs = corpus_probs
        while distance_in_prob_profile(stoc_dedup_probs,
                                       stochastic_dedup_ballot_probs(stoc_dedup_probs)) > 0.00000001 and iters < cap:
            dedup_lt_probs = stochastic_dedup_ballot_probs(dedup_lt_probs)
            iters += 1

        print("Corpus Probs:", round_dict(corpus_probs, 3))
        print("Rejection/Loop Erasure Iterated Equilibrium:", round_dict(rej_le_probs, 3))
        print("Deduplication/Loop Termination Iterated Equilibrium:", round_dict(dedup_lt_probs, 3))
        print("Stochastic Deduplication Iterated Equilibrium:", round_dict(stoc_dedup_probs, 3))
        print("\n")


if __name__ == "__main__":
    BALLOTS = ['A', 'B', 'AB', 'BA']

    mc_match_test_corpi_of_interest()

    iterate_mc_corpi_of_interest()
