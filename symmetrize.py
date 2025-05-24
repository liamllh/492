import numpy as np
from itertools import combinations

from hashmaps import congruency_dict
from chemical_types import PAHLiteral


def pad_tensor(congruency_table: list[list[list[int]]]) -> np.ndarray:

	# find the highest axis of symmetry by checking the first carbon
	highest_symmetry: int = max(
		[len(carbon_symmetry_axes)
		 for carbon_symmetry_axes in congruency_table[0]])

	# pad with None, inplace
	for carbon_symmetry_axes in congruency_table:
		for congruent_carbons in carbon_symmetry_axes:
			padding = highest_symmetry - len(congruent_carbons)
			for _ in range(padding):
				congruent_carbons += [np.nan]

	# convert to array and return
	congruency_array = np.array(congruency_table)
	return congruency_array


def get_substitutions(molecule: PAHLiteral) -> list[dict[np.array: np.array]]:
	# retrieve molecule congruency and pad
	congruency_table = congruency_dict.get(molecule)
	congruency_array = pad_tensor(congruency_table)
	# infer information about molecule from congruency array shape
	n_carbons = np.shape(congruency_array)[0]
	carbon_idxs: list[int] = list(range(n_carbons))

	result: list[dict[np.array: np.array]] = list()

	# perform the symmetry procedure for between 1 and 3 substituents
	for n_subs in range(1, 4):
		all_substitution_patterns = list(combinations(carbon_idxs, n_subs))
		seen_substitutional_patterns_dict: dict[np.array: np.array] = dict()

		for substitutional_pattern in all_substitution_patterns:

			seen = pattern_has_been_seen(substitutional_pattern, list(seen_substitutional_patterns_dict.values()))
			if seen:
				continue

			substitutional_pattern_arr = np.array(substitutional_pattern)
			# transpose to arrange congruent patterns across the lowest axis
			congruent_patterns = congruency_array[substitutional_pattern_arr].transpose()
			# remove nans corresponding to padding
			congruent_patterns = congruent_patterns[~np.isnan(congruent_patterns)]
			# reshape because dropping nans like this collapses axes, congruent sequences still across lowest axis
			congruent_patterns = congruent_patterns.reshape(np.size(congruent_patterns) // n_subs, n_subs)
			# append the substitutional pattern used to create these congruent patterns
			congruent_patterns = np.vstack((substitutional_pattern_arr, congruent_patterns))
			# remember this pattern
			seen_substitutional_patterns_dict.update({substitutional_pattern: congruent_patterns})

		result.append(seen_substitutional_patterns_dict)

	return result


def pattern_has_been_seen(current_pattern: tuple[int], seen_pattern_arr: list[np.array]) -> bool:

	# sequences go across the lowest level axis
	for this_seen_pattern_arr in seen_pattern_arr:

		for idx, pattern in enumerate(this_seen_pattern_arr):

			carbons_in_pattern: list[bool] = [c in pattern for c in current_pattern]
			pattern_is_equivalent_to_current: bool = all(carbons_in_pattern)
			if pattern_is_equivalent_to_current:
				return True

	return False


if __name__ == '__main__':
	for pah in congruency_dict.keys():
		print(pah)
		r = get_substitutions(pah)
		for n in r:
			print(len(n.keys()))
