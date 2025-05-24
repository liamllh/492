import copy
import shutil
from math import exp
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np
from typing import get_args
from pathlib import Path
from shutil import copy
import torch
from torch import Tensor

from chemical_types import PAHLiteral, HalogenLiteral
from hashmaps import congruency_dict, correspondence_dict, atomic_label_number_dict, atomic_label_rgba_dict, \
	atomic_label_weight_dict, base_pah_filepath_dict
from parser import load_gjf_segments, parse_gjf_vector, parse_log
from sample import take_samples
from directories import sep


def pad_congruency(congruency_table: list[list[list[int]]]):
	"""
	Pads the congruency table s.t. bottom level entries are all the same size, allowing the table to be cast as array
	:param congruency_table: Table matching the indices of each substitutable carbon to equivlent carbons by symmetry
	:return: nan-padded congruency table
	"""

	max_sym_ax = max([len(x) for x in congruency_table[0]])
	for atom_congruencies in congruency_table:
		for sym_ax in atom_congruencies:
			n_padding = max_sym_ax - len(sym_ax)
			for _ in range(n_padding):
				sym_ax.append(np.nan)

	return congruency_table


def parse_job(job_list: list[str]) -> tuple[list[str], list[str], str, int, int]:
	"""
	Parse the job section of a GJF
	:param job_list: list of lines in the job section of the GJF
	:return: list of job hyperparameters, list of job parameters, title card, net charge, spin multiplicity
	"""
	gjf_kwargs: list[str] = list()
	gjf_job_args: list[str]

	idx = 0

	# kwargs are first, denoted by leading '%'
	while job_list[idx][0] == "%":
		gjf_kwargs.append(job_list[idx][1:-1])
		idx += 1
	# when we break out of this loop, idx points to the line containing job args

	# ensure that the line after kwargs is job args
	assert job_list[idx][0] == "#", f"Missing job arguments! Got line {job_list[idx]}."
	# skip first two chars to drop "# "; skip last to drop "\n"; split remaining as job args
	gjf_job_args = job_list[idx][2:-1].split(" ")

	# next line is newline, next line is title card, next line is newline, next line is "{charge} {multiplicity}\n"
	# job_list[idx + 1]     job_list[idx + 2]        job_list[idx + 3]     job_list[idx + 4]

	title_card: str = job_list[idx + 2][:-1]  # drop lagging "\n"

	charge, multiplicity = job_list[idx + 4].split(" ")
	# by converting to int here we implicitly assert that the last lines were parsed correctly
	charge, multiplicity = int(charge), int(multiplicity)

	return gjf_kwargs, gjf_job_args, title_card, charge, multiplicity


def parse_connection_full(cx_line: str) -> tuple[int, list[list[int, float]]]:
	"""
	Returns indices of lines corresponding to all the atoms bonded to the atom specified by cx_line
	:param cx_line: of the form 1 2 1.0 3 1.5, where the first item is the current atom index and the following items
		are pairs of (bonded atom, corresponding bond order) for all atoms bonded to the current atom
	:return:
	"""

	parsed_line = list(filter(None, cx_line.split(" ")))
	atom_idx, bonded_idxs, bond_orders = parsed_line[0], parsed_line[1::2], parsed_line[2::2]

	# convert from str and subtract 1 to convert from gjf idx to python idx
	atom_idx = int(atom_idx) - 1
	# subtract 1 to convert from gjf idx to python idx
	bonded_idxs = [int(bonded_idx) - 1 for bonded_idx in bonded_idxs]
	# convert from str
	bond_orders = [float(bond_order) for bond_order in bond_orders]

	bond_idx_order_pairs = [[bonded_idxs[i], bond_orders[i]] for i in range(len(bonded_idxs))]

	return atom_idx, bond_idx_order_pairs


def build_distance_matrix(xyz: list[tuple[float, float, float]]) -> np.array:

	n_atoms = len(xyz)
	dx_matrix = np.zeros((n_atoms, n_atoms))

	position_vectors = np.zeros((n_atoms, 3))
	for idx, line in enumerate(xyz):
		position_vectors[idx, :] = np.array(line)

	for idx, vec in enumerate(position_vectors):
		diffs = vec - position_vectors[idx:]
		dxs = np.sqrt(np.sum(diffs**2, axis=1))
		dx_matrix[idx, idx:] = dxs
		# dx_matrix[idx:, idx] = dxs  # top half of the matrix; redundant

	return dx_matrix


def build_connection_matrix(connection: list[str]):

	n_atoms = len(connection)
	cx_matrix = np.eye(n_atoms)

	for idx, line in enumerate(connection):
		bonded_atoms = parse_connection_full(line)
		cx_matrix[idx, bonded_atoms] = 1
		cx_matrix[bonded_atoms, idx] = 1  # top half of the matrix, redundant

	return cx_matrix


def reduced_mass(mass_0: float, mass_1: float) -> float:
	"""
	µ = (m0 * m1) / (m0 + m1)
	"""
	return mass_0 * mass_1 / (mass_0 + mass_1)


def initialize_hashmaps(xyz: list[str], connection: list[str]):
	"""
	Static function for initializing hashmaps, for use by GJF class
	:param xyz: unparsed xyz section of .gjf
	:param connection: unparsed connection section of .gjf
	:return: atom label hashmap, atom coordinate hashmap, and connection hashmap, all with the same key
	"""
	atom_dict: dict[int: str] = dict()
	coordinate_dict: dict[int: list[float, float, float]] = dict()
	connection_dict: dict[int: list[list[[int, float]]]] = dict()

	# parse coordinate and connection matrices
	for idx, xyz_line in enumerate(xyz):
		atom, (x, y, z) = parse_gjf_vector(xyz_line)
		atom_dict.update({idx: atom})
		coordinate_dict.update({idx: (x, y, z)})

		cx_line = connection[idx]
		atom_idx, bond_idx_order_pairs = parse_connection_full(cx_line)
		connection_dict.update({idx: bond_idx_order_pairs})

	return atom_dict, coordinate_dict, connection_dict


def parse_log_line(line: str) -> tuple[int, int, tuple[float, float, float]]:
	"""
	Returns the atom index, atomic number, and xyz coordinates of one line of the geometry section of a log file
	"""

	sections = list(filter(None, line.split(" ")))
	atom_idx_str, atom_number_str, atom_type_str, x_str, y_str, z_str = sections
	atom_idx = int(atom_idx_str)
	atom_number = int(atom_number_str)
	x = float(x_str)
	y = float(y_str)
	z = float(z_str)

	return atom_idx, atom_number, (x, y, z)


def parse_log_full(log_dir: str) -> tuple[dict[int: dict[int: tuple[float, float, float]]], dict[int: int]]:
	"""
	Parses gaussian 16-generated .log file for coordinates at each SCF iteration and atomic numbers
	:param log_dir: Path to the log file
	:return: [0] dict of SCF iteration numbers mapped to the corresponding geometries at each iteration, which is itself
		a dict of atom indices to tuples of floats, corresponding to xyz coordinates for this SCF iteration;
		[1] dict of atom indices in the log file to their corresponding atomic numbers
	"""
	with open(log_dir, "r") as logfile:

		log_content = logfile.readlines()
		n_atoms = 0

		for line in log_content:
			if "NAtoms" in line:
				# extract number of atoms
				params = list(filter(None, line.split(" ")))
				# implicit error check, ensure that the found parameter can be cast to int
				n_atoms = int(params[1])

		assert n_atoms > 0, f"Error in log {log_dir}. No atoms found."

		coordinate_iterations_raw: list[list[str]] = list()
		for idx, line in enumerate(log_content):
			if "Input orientation" in line:
				# next 4 lines are table header, next n_atoms lines are coordinate information
				dx = log_content[idx + 4: idx + 4 + n_atoms]
				coordinate_iterations_raw.append(dx)

		coordinate_iteration_dict: dict[int: dict[int: tuple[float, float, float]]] = dict()
		for idx, coordinate_iteration_raw in enumerate(coordinate_iterations_raw):
			coordinate_tuples: list[tuple[float, float, float]] = list()
			atom_numbers: list[int] = list()

			for line_entry in coordinate_iteration_raw:
				atom_idx, atom_number, coords = parse_log_line(line_entry)
				atom_numbers.append(atom_number)
				coordinate_tuples.append(coords)

			coordinate_iteration_dict.update({idx: {jdx: coord for jdx, coord in enumerate(coordinate_tuples)}})

		atom_no_dict: dict[int: int] = {kdx: atom_no for kdx, atom_no in enumerate(atom_numbers)}

	return coordinate_iteration_dict, atom_no_dict


class GJF(object):

	def __init__(
			self,
			gjf_path: str,
	):
		# initialize filename-derived properties
		self.gjf_path: str = gjf_path
		self.fname: str = self.gjf_path.split(sep)[-1].split(".")[0]  # filename minus suffix

		# logic for parsing base relaxation jobs vs solvated relaxation jobs
		if "water" in self.gjf_path or "octanol" in self.gjf_path:
			self.pah_base: str = self.fname.split("_")[-2]  # filename minus suffix
		else:
			self.pah_base: str = self.fname.split("_")[-1]  # just the PAH base part

		assert self.pah_base in get_args(PAHLiteral), f"Error parsing PAH base: got {self.pah_base}, which is not one of {PAHLiteral}"

		# load gjf
		job, xyz, connection = load_gjf_segments(gjf_path)
		# parse job
		self.job_kwargs, self.job_args, self.title_card, self.net_charge, self.spin_multiplicity = parse_job(job)
		# initialize hashmaps
		self.atom_dict, self.coordinate_dict, self.connection_dict = initialize_hashmaps(xyz, connection)
		# identify allowed carbons to substitute
		self.bond_orders_to_carbon_dict: dict[int: float] = self.get_carbon_bond_orders()
		# note that this correspondence hashmap breaks if the molecule is a derivative with different indexing than the
		# base PAH files used to generate correspondence dicts
		# this dict is {substitutable carbon idx: corresponding substituted atom}
		self.pah_correspondence_dict: dict[int: int] = correspondence_dict[self.pah_base]
		# maps substitutable carbon index in the correspondence dict to the actual index of that carbon in this molecule
		self.this_correspondence_dict: dict[int: int] = self.generate_this_correspondence_dict()

		# for until I fix reindexing w.r.t. the correspondence hashmaps; will prohibit generation of isomers after self.reindex() has been called
		self.is_reindexed = False

		# for determining whether a molecule was parsed from an original file or generated using GJF.generate_congruent_isomers()
		self.derived_from: str | None = None if "derived from " not in self.title_card else self.title_card[len("derived from "):]

		return

	def __repr__(self):
		ret = (
			f"Molecule at {self.gjf_path} ({'derived from ' + self.derived_from if self.derived_from is not None else 'generated from ' + self.gjf_path}).\n"
			f"This molecule {'has' if self.is_reindexed else 'has not'} been reindexed, and {'cannot' if self.is_reindexed else 'can'} be used to generate new isomers of itself."
		)
		return ret

	def generate_this_correspondence_dict(self):
		"""
		Extracts carbon atoms corresponding to the indices used to create the correspondence hashmap.
		:return:
		"""

		substitutable_atom_idxs = self.pah_correspondence_dict.values()
		this_correspondence_dict: dict[int: int] = dict()

		for atom_idx, atom_cx in self.connection_dict.items():
			# if the atom is carbon
			if self.atom_dict[atom_idx] == "C":
				# and it is bonded to a substitutable atom
				for bonded_atom_idx, bond_order in atom_cx:
					if bonded_atom_idx in substitutable_atom_idxs:
						correspondence_dict_atom_idx = next((k for k, v in self.pah_correspondence_dict.items() if v == bonded_atom_idx), None)
						# remember which atom
						this_correspondence_dict.update({correspondence_dict_atom_idx: atom_idx})
						break

		return this_correspondence_dict

	# need to typehint 'GJF' as str because classes are not lazily evaluated
	def generate_congruent_isomers(self) -> list['GJF']:
		"""
		Generates copies of this object with halogens reindexed to all congruent isomers.
		:return:
		"""

		assert not self.is_reindexed, "Cannot make substitutions from correspondence dict after reindexing!"

		substituted_carbons: list[int] = [int(i) for i in self.fname.split("_")[0].split("-")]
		substituted_carbon_arr: np.array = np.array(substituted_carbons)
		n_subs: int = len(substituted_carbons)
		# we have to pad with nan in case of a single higher symmetry axis, specifically for triphenyl
		congruency_arr = np.array(pad_congruency(congruency_dict.get(self.pah_base)))

		new_gjfs: list[GJF] = list()

		# transpose to arrange congruent patterns across the lowest axis
		congruent_patterns = congruency_arr[substituted_carbon_arr].transpose()
		# remove nans corresponding to padding
		congruent_patterns = congruent_patterns[~np.isnan(congruent_patterns)]
		# reshape because dropping nans like this collapses axes, congruent sequences still across lowest axis
		congruent_patterns = congruent_patterns.reshape(np.size(congruent_patterns) // n_subs, n_subs)
		# append the substitutional pattern used to create these congruent patterns to this array
		congruent_patterns = np.vstack((substituted_carbon_arr, congruent_patterns)).astype(int)

		# convert these to the carbons that need to be substituted using the correspondence dict
		congruent_patterns_reindexed_to_carbon = [[self.this_correspondence_dict.get(atom_idx) for atom_idx in congruent_pattern] for congruent_pattern in congruent_patterns]
		# extract the hydrogen atoms attached to these carbons
		congruent_patterns_substituted_atoms = [[self.get_non_carbon_substituent_idxs(atom_idx) for atom_idx in congruent_pattern] for congruent_pattern in congruent_patterns_reindexed_to_carbon]

		# make new gjf files according to these patterns
		# skip first because this is the one we have currently
		for pattern_num, congruent_substitutional_pattern in enumerate(congruent_patterns_substituted_atoms[1:], 1):

			# make a copy of this gjf and all its attributes
			new_gjf = copy.deepcopy(self)

			# delete old halogenated substitutions
			new_gjf_halogen_substituent_idxs = [i for i, atom in enumerate(new_gjf.atom_dict.values()) if atom in get_args(HalogenLiteral)]
			for idx_to_substitute in new_gjf_halogen_substituent_idxs:
				new_gjf.substitute(idx_to_substitute, "H")

			# parse filename for halogen substituent id
			halogen_id = "Br" if "bromo" in self.fname.split("_")[1] else "Cl"

			# make new halogen substitutions
			for idx_to_substitute in congruent_substitutional_pattern:
				new_gjf.substitute(idx_to_substitute, halogen_id)

			# edit name and path to match reindexing, using the correspondence dict
			# generate new hyphen-separated str of substituted carbon idxs and extract the rest of the name from the current GJF
			new_gjf.fname = "-".join(congruent_patterns[pattern_num].astype(str)) + "_" + "_".join(self.fname.split("_")[1:])
			# construct the path accordingly
			new_gjf.gjf_path = sep.join(self.gjf_path.split(sep)[:-1]) + sep + new_gjf.fname + ".gjf"
			# this enables parsing via the attribute new_gjf.derived_from
			new_gjf.title_card = f"derived from {self.gjf_path}"
			new_gjf.derived_from = self.gjf_path
			# save it
			new_gjfs.append(new_gjf)

		return new_gjfs

	def get_non_carbon_substituent_idxs(self, idx: int) -> int:
		"""
		Returns the indices of non-carbon atoms bonded to the carbon atom at idx. See comments.
		:param idx:
		:return:
		"""
		atom_cx = self.connection_dict.get(idx)
		non_c_bonded_atom_idxs = [bonded_atom_idx for bonded_atom_idx, bond_order in atom_cx if self.atom_dict.get(bonded_atom_idx) != "C"]
		# note that only returning the first idx introduces unpredictable behavior, but for PAH systems there can only
		# be 1 or 0 non-carbon substituents
		# this also implicitly asserts that we did in fact choose a substitutable carbon, otherwise we would be indexing
		# an empty list
		return non_c_bonded_atom_idxs[0]

	def carbon_is_substitutable(self, carbon_idx: int) -> bool:
		"""
		Returns True if the atom at carbon_idx is carbon and is bonded to a hydrogen atom; returns False if not.
		Raises AssertionError if the atom at carbon_idx is not carbon.
		:param carbon_idx: index in the GJF hashmap of the carbon to check
		"""
		assert self.atom_dict[carbon_idx] == "C", f"This atom is not carbon! Found atom {self.atom_dict[carbon_idx]} at index {carbon_idx}."
		return any([self.atom_dict[bonded_idx] == "H" for bonded_idx, order in self.connection_dict[carbon_idx]])

	def get_carbon_bond_orders(self) -> dict[int: float]:
		"""
		Returns a hashmap of multiplicity of bonds of each atom to carbon atoms
		"""

		idxs = list(self.connection_dict.keys())
		bond_order_dict: dict[int: float] = {idx: 0.0 for idx in idxs}

		for atom_idx, atom_cx in self.connection_dict.items():
			for bonded_atom_idx, bond_order in atom_cx:
				if self.atom_dict[bonded_atom_idx] == "C":
					new_idx_order = bond_order_dict[atom_idx] + bond_order
					new_bonded_idx_order = bond_order_dict[bonded_atom_idx] + bond_order
					bond_order_dict.update({atom_idx: new_idx_order})
					bond_order_dict.update({bonded_atom_idx: new_bonded_idx_order})

		return bond_order_dict

	def substitute(self, substituted_atom_idx: int, substituted_atom_label: str):
		"""
		Substitutes the atom at substituted_atom_idx for a different atom defined by substituted_atom_label
		"""
		self.atom_dict.update({substituted_atom_idx: substituted_atom_label})
		return

	def reindex(self, this_index: int, that_index: int):
		"""
		Reindexes this GJF's hashmap values of this_index to the values of that_index, and vice versa
		"""
		assert this_index != that_index, f"Reindexed atoms must be different! Got both {this_index}."

		this_atom, that_atom = self.atom_dict.get(this_index), self.atom_dict.get(that_index)
		assert this_atom is not None, f"No atom at index {this_index}!"
		assert that_atom is not None, f"No atom at index {that_index}!"

		# by setting this flag we prevent generating isomers from indices that are misaligned with the congruency table
		# (because self.generate_congruent_isomers only runs when self.is_redindexed is False)
		self.is_reindexed = True

		# retrieve coordinate and connection info for both atoms
		this_xyz, that_xyz = self.coordinate_dict.get(this_index), self.coordinate_dict.get(that_index)
		this_cx, that_cx = self.connection_dict.get(this_index), self.connection_dict.get(that_index)

		# update hashmaps to swap the keys of this_index with the values of that_index and vice versa
		self.atom_dict.update({this_index: that_atom})
		self.atom_dict.update({that_index: this_atom})
		self.coordinate_dict.update({this_index: that_xyz})
		self.coordinate_dict.update({that_index: this_xyz})
		self.connection_dict.update({this_index: that_cx})
		self.connection_dict.update({that_index: this_cx})

		# !!! the following block does not work as intended !!!
		# if this atom is substitutable
		if this_index in self.this_correspondence_dict.values():
			corresponding_idx = next((k for k, v in self.pah_correspondence_dict.items() if v == this_index), None)
			self.this_correspondence_dict.update({corresponding_idx: that_index})
		# repeat for vice versa
		if that_index in self.this_correspondence_dict.values():
			corresponding_idx = next((k for k, v in self.pah_correspondence_dict.items() if v == this_index), None)
			self.this_correspondence_dict.update({corresponding_idx: this_index})

		# update idxs in connection section if any are bonded to this_atom or that_atom
		for cx_idx, cx_line in self.connection_dict.items():
			reindexed_cx_line: list[list[int, float]] = list()
			needs_update: bool = False

			for bonded_atom_idx, bond_order in cx_line:

				if bonded_atom_idx == this_index:
					bonded_atom_idx = that_index
					needs_update = True
				elif bonded_atom_idx == that_index:
					bonded_atom_idx = this_index
					needs_update = True
				reindexed_cx_line.append([bonded_atom_idx, bond_order])

			# if we changed anything, update it in the hashmap
			if needs_update:
				self.connection_dict.update({cx_idx: reindexed_cx_line})

		return

	def reorder_c_first(self):
		"""
		Reindexes hashmaps to have all carbons in lowest-numbered indices
		"""
		atom_labels = list(self.atom_dict.values())
		n_carbons = atom_labels.count("C")
		c_idxs: list[int] = list()
		non_c_idxs: list[int] = list()

		for i, label in enumerate(atom_labels):
			if label == "C":
				c_idxs.append(i)
			else:
				non_c_idxs.append(i)

		# generate reindexing pairs to result in contiguous block of carbons at start
		non_c_idxs_in_c_block = [i for i in non_c_idxs if i < n_carbons]
		c_idxs_not_in_c_block = [j for j in c_idxs if j >= n_carbons]
		assert len(non_c_idxs_in_c_block) == len(c_idxs_not_in_c_block), "Error in reordering C-first"

		swap_pairs = [[non_c_idxs_in_c_block[k], c_idxs_not_in_c_block[k]] for k in range(len(non_c_idxs_in_c_block))]
		for pair in swap_pairs:
			self.reindex(*pair)

		return

	def write(self, filepath_out: str | None = None, testing_mode: bool = False):
		"""
		Generates new GJF string based on current hashmap states
		:param filepath_out: filepath to write GJF to
		:param testing_mode:
			If testing_mode is False:
				Writes the current state of all hashmaps to a formatted gjf file at filepath_out, if it is specified, or
				self.gjf_path if not
			If testing_mode is True:
				Prints the current state of all hashmaps as a formatted gjf file
		"""

		gjf_out: str = ""

		# job kwargs
		for gjf_kwarg in self.job_kwargs:
			gjf_out += f"%{gjf_kwarg}\n"

		# job args
		gjf_out += f"# {' '.join(self.job_args)}\n"

		# newlines before and after title card
		gjf_out += "\n"
		gjf_out += self.title_card
		gjf_out += "\n\n"

		# charge and spin multiplicity
		gjf_out += f"{self.net_charge} {self.spin_multiplicity}\n"

		# need to save as variables because we cant have \ chars in variable parts of fstring
		two_tabs = "\t\t"
		# atoms and coordinates
		for idx in self.atom_dict.keys():
			# we have to use str(format(x)) to ensure scientific notation is not carried over to gjf, because Gaussian cannot
			# differentiate between scientific notation and integers
			next_xyz = f" {self.atom_dict[idx]}\t\t{two_tabs.join([str(format(x, 'f')) for x in self.coordinate_dict.get(idx)])}\n"
			assert "e" not in next_xyz, f"Error in writing xyz; one or more coordinates likely in scientific notation (line {idx}: {next_xyz})"
			gjf_out += next_xyz

		# split coordinate and connection sections
		gjf_out += "\n"

		# connections
		for idx in self.atom_dict.keys():
			next_cx = f" {idx + 1} {' '.join(str(cx) for cx in chain.from_iterable(self.connection_dict.get(idx)))}\n"
			assert "e" not in next_cx, "Error in writing xyz; one or more coordinates likely in scientific notation"
			gjf_out += next_cx

		if testing_mode:
			print(gjf_out)
			return

		if filepath_out is None:
			filepath_out = self.gjf_path

		# make sure the user is aware when overwriting the file used to generate this GJF object
		if Path(self.gjf_path).exists():
			overwrite_answer = input(f"Are you sure you want to overwrite the gjf file {self.gjf_path}? (y/n)")
			if overwrite_answer == "y":
				print(f"Overwriting {self.gjf_path}")
				pass
			else:
				return

		# save it
		with open(filepath_out, "w") as gjf:
			gjf.write(gjf_out)

		return

	def vectorize_bonds(self) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
		"""
		For drawing bonds in 3d plots
		returns 7 lists, corresponding to:
		 	0, 1, 2	: x, y, z coordinates of atom 0;
		 	3, 4, 5	: x, y, z directions of the vector pointing from atom 0 to atom 1;
		 	6		: bond order between atom 0 and atom 1
		"""
		xs, ys, zs, us, vs, ws, orders = list(), list(), list(), list(), list(), list(), list()

		for atom_0, cx in self.connection_dict.items():
			atom_0_x, atom_0_y, atom_0_z = self.coordinate_dict.get(atom_0)

			for atom_1, bond_order in cx:
				atom_1_x, atom_1_y, atom_1_z = self.coordinate_dict.get(atom_1)
				u, v, w = atom_1_x - atom_0_x, atom_1_y - atom_0_y, atom_1_z - atom_0_z
				xs.append(atom_0_x)
				ys.append(atom_0_y)
				zs.append(atom_0_z)
				us.append(u)
				vs.append(v)
				ws.append(w)
				orders.append(bond_order)

		return xs, ys, zs, us, vs, ws, orders

	def plot(self, title: str | None = None, true_perspective: bool = True, atom_index_labels: bool = False):
		"""
		Plots the atoms in the gjf file according to cartesian coordinates, with bonds drawn according to connection information
		:param title: optional chart title; defaults to self.fname.
		:param true_perspective: if True, forces all axes to be same length to prevent contraction/expansion in one
			dimension. Particularly noticeable for aromatic molecules, where small nonplanar deviations in one
			plane will be magnified if true_perspective is False. Default True.
		:param atom_index_labels: if True, draws atom indices in the gjf on the atoms in the plot. Analogous to
			g16 < view < labels < show labels. Default False.
		:return:
		"""

		atoms = self.atom_dict.values()
		atom_coordinates = self.coordinate_dict.values()
		xs = [xyz[0] for xyz in atom_coordinates]
		ys = [xyz[1] for xyz in atom_coordinates]
		zs = [xyz[2] for xyz in atom_coordinates]
		atomic_numbers = [atomic_label_number_dict.get(a) for a in atoms]
		# scale for viewing
		point_sizes = [40 * (z + 6)**0.5 for z in atomic_numbers]
		# get colors
		color_list = [atomic_label_rgba_dict.get(a) for a in atoms]

		# init plot
		fig = plt.figure()
		ax = fig.add_subplot(**{"projection": "3d"})

		# place atoms
		ax.scatter(xs=xs, ys=ys, zs=zs, s=point_sizes, c=color_list)

		if atom_index_labels:
			for i in range(len(xs)):
				ax.text(x=xs[i], y=ys[i], z=zs[i], s=str(list(self.atom_dict.keys())[i]))

		# draw bonds
		# we use "_" suffix to differentiate from above xs, because some atoms are not in the connection matrix, which
		# avoids redundancies by only using one way bonds (i.e. if 1 and 2 are bonded, only 1 -> 2 will be recorded,
		# because 2 -> 1 is implied)
		xs_, ys_, zs_, us, vs, ws, orders = self.vectorize_bonds()
		# set displayed bond thickness based on bond order
		linewidths = [2 * o for o in orders]
		ax.quiver(xs_, ys_, zs_, us, vs, ws, linewidth=linewidths, arrow_length_ratio=0, color=(0, 0, 0, 0.5))

		# rescale if desired
		if true_perspective:
			# get max axis length
			maxx, minx, maxy, miny, maxz, minz = max(xs), min(xs), max(ys), min(ys), max(zs), min(zs)
			x_rng, y_rng, z_rng = maxx - minx, maxy - miny, maxz - minz
			half_ax_length: float = max((x_rng, y_rng, z_rng)) * 0.5
			# compute midpoints of existing axes
			x_mid, y_mid, z_mid = 0.5 * (maxx + minx), 0.5 * (maxy + miny), 0.5 * (maxz + minz)
			# expand each axis half the length of the max axis length to ensure all axes are same size in real space
			ax.set_xlim(x_mid - half_ax_length, x_mid + half_ax_length)
			ax.set_ylim(y_mid - half_ax_length, y_mid + half_ax_length)
			ax.set_zlim(z_mid - half_ax_length, z_mid + half_ax_length)

		# format
		title_str: str = title if title is not None else self.fname
		ax.set_title(title_str)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

		plt.show()

		return

	def get_normalized_distance_matrix(self) -> Tensor:
		"""
		Returns distance matrix, with values scaled to [-0.5, 0.5]
		:return:
		"""
		dx_nonorm = build_distance_matrix(list(self.coordinate_dict.values()))
		dx_norm = dx_nonorm / np.max(dx_nonorm) - 0.5
		return Tensor(dx_norm)

	def get_connection_matrix(self):
		"""
		Gets reduced-mass weighted connection matrix
		:return:
		"""
		atom_labels = self.atom_dict.values()
		n_atoms = len(atom_labels)
		atom_weights = [atomic_label_weight_dict.get(label) for label in atom_labels]
		# precompute self-reduced masses on the diagonal, equal to half the initial mass
		# negative value encodes self; positive encodes bonded atom
		cx_matrix = -1 * np.eye(n_atoms) * np.array([w * 0.5 for w in atom_weights])

		for idx, atom_bond_order_pairs in enumerate(self.connection_dict.values()):

			for bonded_atom_idx, order in atom_bond_order_pairs:
				# take sqrt to scale while maintaining ordering
				mu = reduced_mass(atom_weights[idx], atom_weights[bonded_atom_idx])
				cx_matrix[idx, bonded_atom_idx] = mu
				cx_matrix[bonded_atom_idx, idx] = mu  # top half of the matrix, redundant

		return Tensor(cx_matrix)

	def get_bond_order_matrix(self) -> Tensor:

		atom_labels = self.atom_dict.values()
		n_atoms = len(atom_labels)
		atom_weights = [atomic_label_weight_dict.get(label) for label in atom_labels]
		# precompute self-reduced masses on the diagonal, equal to half the initial mass
		# negative value encodes self; positive encodes bonded atom
		bx_matrix = -1 * np.eye(n_atoms)

		for idx, atom_bond_order_pairs in enumerate(self.connection_dict.values()):

			for bonded_atom_idx, order in atom_bond_order_pairs:
				bx_matrix[idx, bonded_atom_idx] = order
				# bx_matrix[bonded_atom_idx, idx] = order  # top half of the matrix, redundant

		return Tensor(bx_matrix)


def make_init_derivative_opt_jobs():
	"""
	Creates halogen-substituted PAH bases for project and saves as gjf files for optimization calculations
	:return:
	"""
	remake_init_opt_dirs()

	def update_fname_for_substituted_molecule(substituents: list[int], x_id: HalogenLiteral, base: PAHLiteral):
		x_prefix = '-'.join([str(sub) for sub in substituents])
		x = "bromo" if x_id == "Br" else "chloro"
		return x_prefix + "_" + x + "_" + base

	for pah in get_args(PAHLiteral):

		base_model = GJF(base_pah_filepath_dict.get(pah))
		cdict = correspondence_dict.get(pah)
		samples_all_subs_cl = take_samples(pah)
		samples_all_subs_br = take_samples(pah)

		for n_subs in samples_all_subs_cl:

			for subs in n_subs:

				subs_model = copy.deepcopy(base_model)
				subs_model.fname = update_fname_for_substituted_molecule(subs, "Cl", pah)
				subs_model.gjf_path = f".{sep}pah_bases{sep}{pah}_derivative{sep}" + subs_model.fname + ".gjf"
				subs_model.job_kwargs.append(f"chk={subs_model.fname}.chk")
				subs_model.job_args.remove("geom=connectivity")
				subs_model.job_args.append("geom=newredundant")  # helps prevent errors
				subs_model.title_card = subs_model.fname + " initial optimization"

				for substituent in subs:
					subs_model.substitute(cdict.get(substituent), "Cl")

				subs_model.write()

		for n_subs in samples_all_subs_br:

			for subs in n_subs:

				subs_model = copy.deepcopy(base_model)
				subs_model.fname = update_fname_for_substituted_molecule(subs, "Br", pah)
				subs_model.gjf_path = f".{sep}pah_bases{sep}{pah}_derivatives{sep}" + subs_model.fname + ".gjf"
				subs_model.job_kwargs.append(f"chk={subs_model.fname}.chk")
				subs_model.job_args.remove("geom=connectivity")
				subs_model.job_args.append("geom=newredundant")  # helps prevent errors
				subs_model.title_card = subs_model.fname + " initial optimization"

				for substituent in subs:
					subs_model.substitute(cdict.get(substituent), "Br")

				subs_model.write()


def remake_init_opt_dirs():
	"""
	Mostly for file writing testing; remakes all PAH base parent dirs
	:return:
	"""

	for pah in get_args(PAHLiteral):

		try:
			Path.mkdir(Path(f".{sep}pah_bases{sep}{pah}_derivatives{sep}"))
		except WindowsError:
			print(f"skipping {pah} dir because it already exists")

	return


def make_solvation_jobs():

	job_kwargs = ["mem=32GB", "nprocshared=8"]
	job_args = [
		"opt",
		"M11/6-311+g(d,p)",  # flexible functional
		# "empiricaldispersion=gd3",  # for large PAH rings - removed if M11 functional is active because M11 is not dispersion corrected
		"maxdisk=50GB",  # because Minnesota functionals are in use
		"int=grid=ultrafine",  # tight fit
		"scf=xqc",  # limit to 256 iter
		"geom=allcheckpoint",  # use pre-optimized chk file for all features
		# specify STP conditions
		"temperature=298.15",
		"pressure=1.0"
	]

	for pah in get_args(PAHLiteral):

		run_jobs_dir = f".{sep}pah_bases{sep}{pah}_derivatives"
		files = [str(f) for f in Path.iterdir(Path(run_jobs_dir))]
		chk_files = [f for f in files if f[-3:] == "chk"]

		for chk_path in chk_files:
			chk_name = str(chk_path).split(sep)[-1][:-4]

			# update args and kwargs for water and octanol jobs
			this_job_kwargs_water = job_kwargs.copy()
			this_job_kwargs_octanol = job_kwargs.copy()
			# we strip parent directories with chk_name because this will run in the same folder as the gjf
			this_job_kwargs_water.append(f"chk={chk_name}_water.chk")
			this_job_kwargs_octanol.append(f"chk={chk_name}_octanol.chk")
			water_job_path = chk_path[:-4] + "_water.gjf"
			octanol_job_path = chk_path[:-4] + "_octanol.gjf"

			this_job_args_water = job_args.copy()
			this_job_args_octanol = job_args.copy()
			this_job_args_water.append("scrf=(smd,solvent=water)")
			this_job_args_octanol.append("scrf=(smd,solvent=n-octanol,read)")

			# compose job files and save
			water_job: str = ""
			# add kwargs
			for k in this_job_kwargs_water:
				water_job += f"%{k}\n"

			# indicate starting job section
			water_job += "# "
			# add job args
			for a in this_job_args_water:
				water_job += a + " "

			water_job += "\n"  # newline for end of section

			# same for octanol job
			octanol_job: str = ""
			# add kwargs
			for k in this_job_kwargs_octanol:
				octanol_job += f"%{k}\n"

			# indicate starting job section
			octanol_job += "# "
			# add job args
			for a in this_job_args_octanol:
				octanol_job += a + " "

			octanol_job += "\n\n"

			# specify SMD parameters for n-octanol solvent
			octanol_job += (
				"stoichiometry=C8H18O1\n"
				"solventname=n-octanol\n"
				"eps=9.8629\n"  # ε
				"epsinf=1.4279\n"  # n^2
				"hbondacidity=0.37\n"  # α
				"hbondbasicity=0.48\n"  # ß
				"surfacetensionatinterface=39.01\n"  # gamma
				"carbonaromaticity=0.0\n"  # Φ
				"electronegativehalogenicity=0.0\n"  # φ
				"\n\n"  # newline for end of section

				# IGNORE BELOW - for CPCM model, which was not implemented in this study
				# 0.825 g mL^-1 at STP
				# = (0.825 / 130.23) mol mL^-1
				# = 0.00633 mol * 10e-24 A^-3
				# = 0.3815e23 * 10e-24 molecules A^-3
				# = 0.03815 molecules A^-3
				# "density=0.03815\n"
				# we know that volume / molecule = (1 / 0.03815) A^3 / molecule
				# in the HCP packing structure, each molecule is 74% occupied and 26% empty space (assume spherical)
				# => V_occ = V * 0.74 = 0.74 * 26.21 A^3 = 19.40 A^3
				# => r = (3/4π) V_occ^(1/3) = 0.239 * 2.687 = 0.6422
				# => molecular radius = 26.21**(1/3) A = 2.970 A
				# "rsolv=0.6422\n\n"
			)

			# save gjfs
			with open(water_job_path, "w") as water_gjf:
				print(f"writing {water_job_path}")
				water_gjf.write(water_job)

			with open(octanol_job_path, "w") as octanol_gjf:
				print(f"writing {octanol_job_path}")
				octanol_gjf.write(octanol_job)

			# save copies of the chks used to generate here too
			water_chk_path = Path(chk_path[:-4] + "_water.chk")
			if not Path.exists(Path(water_chk_path)):
				shutil.copy(chk_path, water_chk_path)

			octanol_chk_path = Path(chk_path[:-4] + "_octanol.chk")
			if not Path.exists(Path(octanol_chk_path)):
				shutil.copy(chk_path, octanol_chk_path)

	return


def compute_log_kow(water_log_path: str, octanol_log_path: str) -> float:
	"""
	Computes log kow as:
		log(kow) = 2.625e6 * (E_oct - E_water) / RT, where E_oct is molecule energy in n-octanol solvent, E_water is
		molecule energy in water solvent, R is the gas constant (J mol^-1 K^-1), T is 298.15K, and 2.625e6 is a
		conversion constant from Hartrees to J mol^-1
	:param water_log_path: Path to water solvation job log file
	:param octanol_log_path: Path to octanol solvation job log file
	:return: computed log(kow) for this molecule
	"""
	water_energies, water_cycles, water_duration = parse_log(water_log_path)
	water_energy = water_energies[-1]  # in eH
	octanol_energies, octanol_cycles, octanol_duration = parse_log(octanol_log_path)
	octanol_energy = octanol_energies[-1]

	hartree_to_j_mol = 2_625_500
	rt = 8.31 * 298.15
	coeff = hartree_to_j_mol / rt
	# print(water_energy, octanol_energy)
	log_kow = (water_energy - octanol_energy) * coeff

	# print(water_energy, octanol_energy, water_energy - octanol_energy)
	# print([(water_energies[i] - octanol_energies[i]) * coeff for i in range(min(len(water_energies), len(octanol_energies)))])
	return log_kow


def move_chks():
	"""
	Onetime function to move all optimized XPAH chk files from PAH folder to PAH derivatives folder
	"""

	for chk in Path.iterdir(Path(f".{sep}pah_bases{sep}init_opt_chks{sep}")):

		chk_path_str = str(chk)
		parent_pah = chk_path_str.split(sep)[-1].split("_")[-1][:-4]
		destination = f".{sep}pah_bases{sep}{parent_pah}_derivatives" + chk_path_str.split(sep)[-1]
		shutil.copy(chk_path_str, destination)


def randomize_matrix_rows(matrix: Tensor, pattern = None):
	"""
	Reorders rows of matrix randomly, maintining values at the intersections of original row column indices
	:return:
	"""
	n = 50
	dimsize = matrix.shape[0]
	if pattern is not None:
		row_order, column_order = pattern
	else:
		row_order = np.random.choice(np.arange(dimsize), dimsize, replace=False)
		column_order = np.random.choice(np.arange(dimsize), dimsize, replace=False)

	# randomize rows
	matrix = matrix[row_order, :]
	# randomize columns
	matrix = matrix[:, column_order]

	return matrix, (row_order, column_order)


def randomize_matrices(
		bx_init: Tensor,
		cx_init: Tensor,
		dx_init: Tensor,
		n: int,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
	"""
	Randomizes n copies of bx_init, cx_init, and dx_init
	:param bx_init: initial bond order matrix
	:param cx_init: initial connection matrix
	:param dx_init: initial distance matrix
	:param n: number of randomized structures to generate
	:return:
	"""
	bxs: list[Tensor] = list()
	cxs: list[Tensor] = list()
	dxs: list[Tensor] = list()

	for _ in range(n):
		bx_n, pattern = randomize_matrix_rows(bx_init)
		cx_n, _ = randomize_matrix_rows(cx_init, pattern=pattern)
		dx_n, _ = randomize_matrix_rows(dx_init, pattern=pattern)
		bxs.append(bx_n)
		cxs.append(cx_n)
		dxs.append(dx_n)

	return bxs, cxs, dxs


def get_finished_data(
		n_repeats: int = 1
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[float], list[str]]:
	"""
	Iterates over finished file folders and retrieves connection, distance, and log kow values for each run XPAH and
		its symmetrically redundant isomers
	:param n_repeats: number of randomized datapoints to generate for each completed job
	:return: hashmaps of bonding, connection and distance matrices and aligned list of log_kow values
	"""

	all_bx_matrices: list[Tensor] = list()
	all_cx_matrices: list[Tensor] = list()
	all_dx_matrices: list[Tensor] = list()
	all_kows: list[float] = list()
	hashmap_key: list[str] = list()

	for pah in get_args(PAHLiteral):

		finished_job_dir = f".{sep}pah_bases{sep}{pah}_finished{sep}"
		water_jobs = [str(p) for p in Path.iterdir(Path(finished_job_dir)) if "_water.log" in str(p)]

		for water_log in water_jobs:
			# go from format a-b-c_halo_pah_water.log to format a-b-c_halo_pah
			substitutional_pattern_str, halo_prefix, base = water_log.split(sep)[-1].split("_")[:-1]
			xpah_id = f"{substitutional_pattern_str}_{halo_prefix}_{base}"

			octanol_log = water_log[:-9] + "octanol.log"
			base_gjf_dir = f".{sep}pah_bases{sep}{pah}.gjf"
			base_structure = GJF(base_gjf_dir)
			# base_structure.plot(atom_index_labels=True)

			# screen for data outliers
			log_kow = compute_log_kow(water_log, octanol_log)
			if log_kow > 10:
				print(f"abnormal log kow detected: {log_kow} @ {xpah_id} | skipping")
				continue
			if log_kow < 5:
				print(f"abnormal log kow detected: {log_kow} @ {xpah_id} | skipping")
				continue

			# we passed the outlier screen, so save this molecule in the key
			for _ in range(n_repeats):
				hashmap_key.append(xpah_id)

			# because we extract the distance matrix from the unoptimized structure, these distances will not be the
			# lowest energy conformer in the respective solvent. However, the model should be powered entirely by
			# connective data, so this mismatch should not matter.
			base_dx = base_structure.get_normalized_distance_matrix()
			base_cx = base_structure.get_connection_matrix()
			base_bx = base_structure.get_bond_order_matrix()

			base_bxs, base_cxs, base_dxs = randomize_matrices(base_bx, base_cx, base_dx, n_repeats)
			for b in base_bxs:
				all_bx_matrices.append(b)
			for c in base_cxs:
				all_cx_matrices.append(c)
			for d in base_dxs:
				all_dx_matrices.append(d)
			for _ in range(n_repeats):
				all_kows.append(log_kow)

			print(f"{water_log.split(sep)[-1].replace('_water.log', '')} | log(kow) = {log_kow}, kow = {exp(log_kow)}")

	return all_bx_matrices, all_cx_matrices, all_dx_matrices, all_kows, hashmap_key


def zero_pad_matrix(
		matrix: Tensor,
		padding: int,
		front_pad_rows: bool = True,
		front_pad_columns: bool = True,
) -> Tensor:
	"""
	Pads matrix with zero values to give final shape padding x padding
	:param matrix: 2d input Tensor; row and column dims do not need to match
	:param padding: desired dimsize of each dim of the padded output; must be >= the dimsizes of matrix
 	:param front_pad_rows: if True, new matrix will have leading rows be 0s
	:param front_pad_columns: if True, new matrix will have leading columns be 0s
	:return:
	"""

	mat_shape = matrix.shape
	assert len(mat_shape) == 2, f"Input must be a 2d matrix; got {len(mat_shape)} dimensions ({mat_shape})"

	col_dim: int = mat_shape[0]
	row_dim: int = mat_shape[1]
	assert col_dim <= padding, f"Cannot pad a matrix with more columns ({col_dim}) than padding ({padding})"
	assert row_dim <= padding, f"Cannot pad a matrix with more rows ({row_dim}) than padding ({padding})"
	mat_out = torch.zeros((padding, padding))

	if front_pad_columns and front_pad_rows:
		mat_out[:col_dim, :row_dim] = matrix
	elif front_pad_columns and not front_pad_rows:
		mat_out[:col_dim, -row_dim:] = matrix
	elif not front_pad_columns and front_pad_rows:
		mat_out[-col_dim:, :row_dim] = matrix
	else:
		mat_out[-col_dim:, -row_dim:] = matrix

	return mat_out


def zero_pad_input_data(data: list[Tensor]) -> Tensor:
	"""
	Takes list of 2d data with varying shapes, zero-pads and stacks along 0th dimension for batching
	:param data: list of 2d input tensors
	:return:
	"""

	padding = 30  # max n atoms
	tensorized_data_list: list[Tensor] = list()
	for data_arr in data:

		data_tensor = zero_pad_matrix(data_arr, padding)
		tensorized_data_list.append(data_tensor)

	data_collated = torch.stack(tensorized_data_list, dim=0)
	return data_collated


def get_training_data():
	"""
	Saves 512 randomized matrix representations of each molecule as a single tensor for model training
	"""

	bxs, cxs, dxs, kows, molnames = get_finished_data(n_repeats=512)

	bxs_tensor = zero_pad_input_data(bxs)
	cxs_tensor = zero_pad_input_data(cxs)
	dxs_tensor = zero_pad_input_data(dxs)
	kow_tensor = Tensor(kows)

	bxpath = "./training_data/bond_order_matrix_randomized"
	cxpath = "./training_data/connection_matrix_randomized"
	dxpath = "./training_data/distance_matrix_randomized"
	kowpath = "./training_data/kow_vector_randomized"

	print(bxs_tensor.shape, cxs_tensor.shape, dxs_tensor.shape, kow_tensor.shape)
	torch.save(bxs_tensor, bxpath)
	torch.save(cxs_tensor, cxpath)
	torch.save(dxs_tensor, dxpath)
	torch.save(kow_tensor, kowpath)
	with open("./training_data/data_ids.txt", "w") as data_ids:
		data_ids.write("\n".join(molnames))

	return


if __name__ == '__main__':

	get_training_data()
