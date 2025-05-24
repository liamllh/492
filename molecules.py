import numpy as np
from itertools import combinations
from typing import get_args

from chemical_types import PAHLiteral, HalogenLiteral
from hashmaps import congruency_dict
from symmetrize import pad_tensor, pattern_has_been_seen
from hashmaps import filepath_dict, substitution_prefix_dict


class PAHBase(object):

	def __init__(
			self,
			base: PAHLiteral,
	) -> None:

		self.base = base
		# retrieve molecule congruency and pad
		self.congruency_table: list[list[list[int]]] = congruency_dict.get(self.base)
		self.congruency_array: np.array = pad_tensor(self.congruency_table)
		# infer information about molecule from congruency array shape
		self.n_carbons = np.shape(self.congruency_array)[0]
		self.carbon_idxs = list(range(self.n_carbons))
		# initialize symmetry (updated by the self.update_symmetry() method)
		self.symmetry: list[dict[np.array: np.array]] = list()
		return

	def get_all_symmetrically_redundant_molecules(
			self,
			n_subs: int,
	):

		all_substitution_patterns = list(combinations(self.carbon_idxs, n_subs))
		seen_substitutional_patterns_dict: dict[np.array: np.array] = dict()
		# make type checker happy, can't parse np array for dtype
		substitutional_pattern: tuple[int]

		for substitutional_pattern in all_substitution_patterns:

			seen = pattern_has_been_seen(substitutional_pattern, list(seen_substitutional_patterns_dict.values()))
			if seen:
				continue

			congruent_patterns = self.get_symmetrically_redundant_molecules(substitutional_pattern, n_subs)
			# remember this pattern
			seen_substitutional_patterns_dict.update({substitutional_pattern: congruent_patterns})

		return seen_substitutional_patterns_dict

	def get_symmetrically_redundant_molecules(
			self,
			substitutional_pattern: tuple[int,...],
			n_subs: int,
	) -> np.array:

		substitutional_pattern_arr = np.array(substitutional_pattern)
		# transpose to arrange congruent patterns across the lowest axis
		congruent_patterns = self.congruency_array[substitutional_pattern_arr].transpose()
		# remove nans corresponding to padding
		congruent_patterns = congruent_patterns[~np.isnan(congruent_patterns)]
		# reshape because dropping nans like this collapses axes, congruent sequences still across lowest axis
		congruent_patterns = congruent_patterns.reshape(np.size(congruent_patterns) // n_subs, n_subs)
		# append the substitutional pattern used to create these congruent patterns to this array
		congruent_patterns = np.vstack((substitutional_pattern_arr, congruent_patterns))

		return congruent_patterns

	def update_symmetry(
			self,
			n: int,
	) -> None:
		symmetric_representations = self.get_all_symmetrically_redundant_molecules(n)
		self.symmetry.append(symmetric_representations)
		return


class XPAH(PAHBase):

	def __init__(
			self,
			base: PAHLiteral,
			substitutional_pattern: tuple[int,...],
			substituted_atom: HalogenLiteral,
	):
		super().__init__(base)
		self.pah_base_filepath = filepath_dict.get(base)
		self.substitutional_pattern = substitutional_pattern
		self.n_subs = len(self.substitutional_pattern)
		self.substituted_atom = substituted_atom

		pass

	def __repr__(self):
		out = (
			f"{','.join([str(s) for s in self.substitutional_pattern])}-{substitution_prefix_dict.get(self.n_subs)}{'chloro' if self.substituted_atom == 'Cl' else 'bromo'}{self.base}\n"
			f"Congrent substitutions: {self.get_symmetrically_redundant_molecules(self.substitutional_pattern, self.n_subs)}"
		)
		return out


def workflow(pah_base: PAHLiteral):

	base_molecule = PAHBase(pah_base)

	for n in range(1, 4):

		base_molecule.update_symmetry(n)

		for unique_symmetry in base_molecule.symmetry[-1]:

			for halogen in get_args(HalogenLiteral):
				xpah = XPAH(pah_base, unique_symmetry, halogen)
				print(xpah)

	return


if __name__ == '__main__':
	workflow("anthracene")
	pass
