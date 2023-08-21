from ballot_generator import *
from profile import PreferenceProfile

from scipy import stats
from pathlib import Path
import pickle
import math


def do_ballot_probs_match_ballot_dist(
		ballot_prob_dict: dict, generated_profile: PreferenceProfile, n: int, alpha=0.95
):
	n_ballots = generated_profile.num_ballots()
	ballot_conf_dict = {
		b: binomial_confidence_interval(p, n_attempts=int(n_ballots), alpha=alpha)
		for b, p in ballot_prob_dict.items()
	}

	failed = 0

	for b in ballot_conf_dict.keys():
		b_list = [{c} for c in b]
		ballot = next(
			(
				element
				for element in generated_profile.ballots
				if element.ranking == b_list
			),
			None,
		)
		ballot_weight = 0
		if ballot is not None:
			ballot_weight = ballot.weight
		if not (
				int(ballot_conf_dict[b][0]) <= ballot_weight <= int(ballot_conf_dict[b][1])
		):
			failed += 1

	n_factorial = math.factorial(n)
	stdev = np.sqrt(n_factorial * alpha * (1 - alpha))
	return failed < (n_factorial * (1 - alpha) + 2 * stdev)


def binomial_confidence_interval(probability, n_attempts, alpha=0.95):
	# Calculate the mean and standard deviation of the binomial distribution
	mean = n_attempts * probability
	std_dev = np.sqrt(n_attempts * probability * (1 - probability))

	z_score = stats.norm.ppf((1 + alpha) / 2)  # Z-score for 99% confidence level
	margin_of_error = z_score * std_dev

	conf_interval = (mean - margin_of_error, mean + margin_of_error)

	return conf_interval


def _test_Cambridge_correctness():
	BASE_DIR = Path(__file__).resolve().parent
	DATA_DIR = BASE_DIR / "data/"
	path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

	candidates = ["W1", "W2", "C1", "C2"]
	ballot_length = None
	slate_to_candidate = {"W": ["W1", "W2"], "C": ["C1", "C2"]}
	pref_interval_by_bloc = {
		"W": {"W1": 0.4, "W2": 0.4, "C1": 0.1, "C2": 0.1},
		"C": {"W1": 0.1, "W2": 0.1, "C1": 0.4, "C2": 0.4},
	}
	bloc_voter_prop = {"W": 0.5, "C": 0.5}
	bloc_crossover_rate = {"W": {"C": 0}, "C": {"W": 0}}

	cs = CambridgeSampler(
		candidates=candidates,
		ballot_length=ballot_length,
		slate_to_candidate=slate_to_candidate,
		pref_interval_by_bloc=pref_interval_by_bloc,
		bloc_voter_prop=bloc_voter_prop,
		bloc_crossover_rate=bloc_crossover_rate,
		path=path,
	)

	with open(path, "rb") as pickle_file:
		ballot_frequencies = pickle.load(pickle_file)
	total_camb = sum(ballot_frequencies.values())
	# [print(k, v / total_camb) for k, v in sorted(ballot_frequencies.items(), key=lambda x: x[1], reverse=True)]

	slates = list(slate_to_candidate.keys())

	# Probability of producting a white leading ballot
	# = p(white) * p(non-crossover) * p(bloc ordering) * p(white slate order for whites) * p(poc slate order for whites)
	# + p(poc) * p(crossover) * p(bloc ordering) * p(white slate ordering for poc) * p(poc slate order for poc)

	# Let's update the running probability of the ballot based on where we are in the nesting
	ballot_prob_dict = dict()
	ballot_prob = [0, 0, 0, 0, 0]
	# p(white) vs p(poc)
	for slate in slates:
		opp_slate = next(iter(set(slates).difference(set(slate))))

		slate_cands = slate_to_candidate[slate]
		opp_cands = slate_to_candidate[opp_slate]

		ballot_prob[0] = bloc_voter_prop[slate]
		prob_ballot_given_slate_first = bloc_order_probs_slate_first(slate, ballot_frequencies)
		# p(crossover) vs p(non-crossover)
		for voter_bloc in slates:
			opp_voter_bloc = next(iter(set(slates).difference(set(voter_bloc))))
			if voter_bloc == slate:
				ballot_prob[1] = 1 - bloc_crossover_rate[voter_bloc][opp_voter_bloc]

				# p(bloc ordering)
				for slate_first_ballot, slate_ballot_prob in prob_ballot_given_slate_first.items():
					ballot_prob[2] = slate_ballot_prob

					# Count number of each slate in the ballot
					slate_ballot_count_dict = {}
					for s, sc in slate_to_candidate.items():
						count = sum([c == s for c in slate_first_ballot])
						slate_ballot_count_dict[s] = min(count, len(sc))

					# Make all possible perms with right number of slate candidates
					slate_perms = list(set([p[:slate_ballot_count_dict[slate]] for p in list(it.permutations(slate_cands))]))
					opp_perms = list(set([p[:slate_ballot_count_dict[opp_slate]] for p in list(it.permutations(opp_cands))]))

					only_slate_interval = {
						c: share
						for c, share in pref_interval_by_bloc[voter_bloc].items()
						if c in slate_cands
					}
					only_opp_interval = {
						c: share
						for c, share in pref_interval_by_bloc[voter_bloc].items()
						if c in opp_cands
					}
					for sp in slate_perms:
						ballot_prob[3] = compute_pl_prob(sp, only_slate_interval)
						for op in opp_perms:
							ballot_prob[4] = compute_pl_prob(op, only_opp_interval)

							# ADD PROB MULT TO DICT
							ordered_slate_cands = list(sp)
							ordered_opp_cands = list(op)
							ballot_ranking = []
							for c in slate_first_ballot:
								if c == slate:
									if ordered_slate_cands:
										ballot_ranking.append(ordered_slate_cands.pop(0))
								else:
									if ordered_opp_cands:
										ballot_ranking.append(ordered_opp_cands.pop(0))
							prob = np.prod(ballot_prob)
							ballot = tuple(ballot_ranking)
							ballot_prob_dict[ballot] = ballot_prob_dict.get(ballot, 0) + prob
			else:
				ballot_prob[1] = bloc_crossover_rate[voter_bloc][opp_voter_bloc]

				# p(bloc ordering)
				for slate_first_ballot, slate_ballot_prob in prob_ballot_given_slate_first.items():
					ballot_prob[2] = slate_ballot_prob

					# Count number of each slate in the ballot
					slate_ballot_count_dict = {}
					for s, sc in slate_to_candidate.items():
						count = sum([c == s for c in slate_first_ballot])
						slate_ballot_count_dict[s] = min(count, len(sc))

					# Make all possible perms with right number of slate candidates
					slate_perms = [p[:slate_ballot_count_dict[slate]] for p in list(it.permutations(slate_cands))]
					opp_perms = [p[:slate_ballot_count_dict[opp_slate]] for p in list(it.permutations(opp_cands))]
					only_slate_interval = {
						c: share
						for c, share in pref_interval_by_bloc[opp_voter_bloc].items()
						if c in slate_cands
					}
					only_opp_interval = {
						c: share
						for c, share in pref_interval_by_bloc[opp_voter_bloc].items()
						if c in opp_cands
					}
					for sp in slate_perms:
						ballot_prob[3] = compute_pl_prob(sp, only_slate_interval)
						for op in opp_perms:
							ballot_prob[4] = compute_pl_prob(op, only_opp_interval)

							# ADD PROB MULT TO DICT
							ordered_slate_cands = list(sp)
							ordered_opp_cands = list(op)
							ballot_ranking = []
							for c in slate_first_ballot:
								if c == slate:
									if ordered_slate_cands:
										ballot_ranking.append(ordered_slate_cands.pop())
								else:
									if ordered_opp_cands:
										ballot_ranking.append(ordered_opp_cands.pop())
							prob = np.prod(ballot_prob)
							ballot = tuple(ballot_ranking)
							ballot_prob_dict[ballot] = ballot_prob_dict.get(ballot, 0) + prob

	# Now see if ballot prob dict is right
	test_profile = cs.generate_profile(num_ballots=10000)
	return do_ballot_probs_match_ballot_dist(ballot_prob_dict=ballot_prob_dict,
											 generated_profile=test_profile,
											 n=len(candidates))


def compute_pl_prob(perm, interval):
	pref_interval = interval.copy()
	prob = 1
	for c in perm:
		if sum(pref_interval.values()) == 0:
			prob *= 1 / np.math.factorial(len(pref_interval))
		else:
			prob *= pref_interval[c] / sum(pref_interval.values())
		del pref_interval[c]
	return prob


def bloc_order_probs_slate_first(slate, ballot_frequencies):
	slate_first_count = sum(
		[
			freq
			for ballot, freq in ballot_frequencies.items()
			if ballot[0] == slate
		]
	)
	prob_ballot_given_slate_first = {
		ballot: freq / slate_first_count
		for ballot, freq in ballot_frequencies.items()
		if ballot[0] == slate
	}
	return prob_ballot_given_slate_first


if __name__ == "__main__":
	print(_test_Cambridge_correctness())
