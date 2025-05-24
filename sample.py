from chemical_types import PAHLiteral
from symmetrize import get_substitutions
from hashmaps import sample_dict
import numpy as np
from random import sample


def take_samples(molecule: PAHLiteral) -> list[np.array]:

	symmetry_dicts = get_substitutions(molecule)
	n_samples = sample_dict.get(molecule)
	chosen_samples: list[np.array] = list()

	for i, sample_size in enumerate(n_samples):

		unique_substitutions: np.array = np.array(list(symmetry_dicts[i].keys()))
		population = unique_substitutions.shape[0]
		# random.sample is nonrepeating; sample space is the allowed row indices of the unique substitution array,
		# 	i.e. the range of its shape along the 0th dim
		sampler = np.array(sample(list(range(population)), sample_size))
		# apply sample across rows
		samples = unique_substitutions[sampler]
		# record chosen samples
		chosen_samples.append(samples)

	return chosen_samples
