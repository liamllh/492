import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hashmaps import atomic_number_label_dict


result_dir: str = "C:\\Users\\lholton\\PycharmProjects\\492\\pah_bases\\"
base_folders: tuple[str, str, str, str, str, ] = (
	"anthracene_derivatives_run\\",
	"phenanthrene_derivatives_run\\",
	"pyrene_derivatives_run\\",
	"triphenyl_derivatives_run\\",
	"benzaanthracene_derivatives_run\\",
)
LOG_ERROR_LINES: int = 5


def parse_cpu_time(job_cpu_time_str: str) -> float:
	"""
	Returns elapsed time in s from Gaussian output format
	:return:
	"""
	job_cpu_time_split = list(filter(None, job_cpu_time_str.split(" ")))

	# ["Job", "cpu", "time:", "d", "days", "h", "hours", "m", "minutes", "s", "seconds"]
	days = int(job_cpu_time_split[3])
	hours = int(job_cpu_time_split[5])
	minutes = int(job_cpu_time_split[7])
	seconds = float(job_cpu_time_split[9])

	duration_s: float = seconds + 60 * minutes + 3600 * hours + 3600 * 24 * days
	return duration_s


def parse_scf_energy(scf_done_str: str) -> tuple[float, int]:
	"""
	Returns SCF energy and cycle for a single iteration
	:param scf_done_str: line starting with "SCF Done:" from the .log file
	:return:
	"""
	split_line = list(filter(None, scf_done_str.split(" ")))
	energy = float(split_line[4])
	n_cycles = int(split_line[7])

	return energy, n_cycles


def parse_log(log_dir: str | Path) -> tuple[list[float], list[int], float]:
	"""
	Parses log file for job info
	:param log_dir:
	:return: lists of energies and number of cycles to obtain energy at each SCF iteration, and the cpu time in s
	"""

	with open(log_dir, "r") as log:

		# last "SCF Done: " is our final energy
		# split by whitespace yields ["SCF", "Done:", "E({method})", "=", "{E value}", "A.U.", "after", "{n}", "cycles"]
		lines = log.readlines()
		if "Normal termination" not in lines[-1]:
			print(f"Abnormal termination detected in {log_dir}. Final {LOG_ERROR_LINES} lines of log: \n{''.join(lines[-LOG_ERROR_LINES:])}")
			return list(), list(), 0.0

		energies: list[float] = list()
		cycles: list[int] = list()

		for line in lines:

			if "SCF Done" in line:
				energy, cycle = parse_scf_energy(line)
				energies.append(energy)
				cycles.append(cycle)

			# todo add the dx matrix parser here

		# if this function is called, we already know that termination was normal, so the cpu time will always be the
		# 4th line from bottom
		duration_s = parse_cpu_time(lines[-4])

		return energies, cycles, duration_s


def get_job_dirs() -> tuple[list[Path], list[Path], list[str]]:

	"""
	Retrieves job files, log files, and filenames from separate XPAH job runs
	:return: list of job Paths, list of log Paths, associated names as strs
	"""
	log_files: list[Path] = list()
	gjf_files: list[Path] = list()
	fnames: list[str] = list()

	for base_folder in base_folders:

		full_path = result_dir + base_folder
		# these are sorted by default, in order [sample_1.gjf, sample_1.log, sample_2.gjf, sample_2.log, ...]
		all_files = [fi for fi in Path.iterdir(Path(full_path))]

		for gjf_file in all_files[::2]:
			gjf_files.append(gjf_file)

		for log_file in all_files[1::2]:
			log_files.append(log_file)
			fnames.append(str(log_file).split("\\")[-1][:-4])

	return gjf_files, log_files, fnames


def load_gjf_segments(job_path) -> tuple[list[str], list[str], list[str]]:
	"""
	Parses the job file for a molecule into sections corresponding to:
		Job
			kwargs, opt + freq, etc
		xyz
			xyz coordinates of each atom in the molecule
			i.e. "C 1.0 1.0 0.5" for a carbon atom at (1, 1, 0.5)
		connectivity
			connectivity strings
			i.e. "0 1 1.0 2 1.5 ..." for each atom

	:param job_path: location of the job file
	:return: lines of each section, each as list of str
	"""

	with open(job_path, "r") as gjf:

		gjf_lines: list[str] = [line for line in gjf.readlines()]
		n_lines: int = len(gjf_lines)

		kwargs_end = 7
		n_atoms: int = (n_lines - kwargs_end - 2) // 2

		xyz_end = kwargs_end + n_atoms
		connectivity_end = xyz_end + n_atoms

		# %mem=32g, opt freq ..., etc.
		kwargs_and_instructions = gjf_lines[:kwargs_end]
		# C       1.0 1.0 1.0, etc
		xyz = gjf_lines[kwargs_end:xyz_end]
		# 1 2 2.0 8 1.5, etc
		connectivity = gjf_lines[xyz_end + 1:connectivity_end + 1]

	return kwargs_and_instructions, xyz, connectivity


def build_connection_matrix(xyz: list[str], connection: list[str], padding: int):

	cx_matrix = np.eye(padding)
	n_atoms = len(xyz)
	padding_start = n_atoms - padding
	# delete identity elements in padded part
	cx_matrix[padding_start:, padding_start:] = 0

	assert padding_start <= 0, f"Incorrect padding: this molecule has {n_atoms} atoms, which is greater than the padding of {padding}."

	for idx, line in enumerate(connection):
		bonded_atoms = parse_connection(line)
		cx_matrix[idx, bonded_atoms] = 1
		# cx_matrix[bonded_atoms, idx] = 1  # top half of the matrix, redundant

	return cx_matrix


def build_distance_matrix(xyz: list[str], padding: int):

	dx_matrix = np.zeros((padding, padding))
	n_atoms = len(xyz)
	padding_start = n_atoms - padding

	assert padding_start <= 0, f"Incorrect padding: this molecule has {n_atoms} atoms, which is greater than the padding of {padding}."

	position_vectors = np.zeros((padding, 3))
	for idx, line in enumerate(xyz):
		atom, (x, y, z) = parse_gjf_vector(line)
		position_vectors[idx, :] = np.array((x, y, z))

	for idx, vec in enumerate(position_vectors):
		diffs = vec - position_vectors[idx:padding_start]
		dxs = np.sqrt(np.sum(diffs**2, axis=1))
		dx_matrix[idx, idx:padding_start] = dxs
		# dx_matrix[idx:padding_start, idx] = dxs  # top half of the matrix; redundant

	return dx_matrix


def parse_connection(cx_line: str) -> list[int]:
	"""
	Returns indices of lines corresponding to all the atoms bonded to the atom specified by cx_line
	:param cx_line: of the form 1 2 1.0 3 1.5, where the first element is the current atom and the following elements
		are sequential pairs of (bonded atom, corresponding bond order) for all atoms bonded to the current atom
	:return:
	"""

	parsed_line = list(filter(None, cx_line.split(" ")))
	atom_idx, bonded_idxs = parsed_line[0], parsed_line[1::2]

	# subtract 1 to go from gjf idx to python idx
	bonded_idxs = [int(bonded_idx) - 1 for bonded_idx in bonded_idxs]
	# todo deprecate this
	return bonded_idxs


def parse_gjf_vector(xyz_line: str) -> tuple[str, tuple[float, float, float]]:

	atom, x, y, z = list(filter(None, xyz_line.split(" ")))
	x = float(x)
	y = float(y)
	z = float(z)

	return atom, (x, y, z)


def build_matrices(gjf_filepath: str, padding: int):

	job, xyz, connection = load_gjf_segments(gjf_filepath)
	connection_matrix = build_connection_matrix(xyz, connection, padding)
	distance_matrix = build_distance_matrix(xyz, padding)

	return connection_matrix, distance_matrix


# todo
def parse_log_distance_matrix(logfile_distance_matrix_section: list[str]):
	"""
	Builds distance matrix from last coordinates in the .log file
	:param logfile_distance_matrix_section:
	:return:
	"""
	# this section is of the form:
	# 				Standard orientation:
	# -----------------------------------------------------
	# Center	Atomic	Atomic		Coordinates (Angstroms)
	# Number	Number	Type		X		Y		Z
	# -----------------------------------------------------
	# 1			6		0			x1		y1		z1
	# 2			6		0			x2		y2		z2
	# ...
	# -----------------------------------------------------

	# drop table title [0], headers [2, 3], spacers [1, 4], and final line spacer [-1]
	atom_lines = logfile_distance_matrix_section[5:-1]

	for atom_line in atom_lines:
		atom_idx_str, atomic_number_str, atomic_type, x_str, y_str, z_str = list(filter(None, atom_line.split(" ")))
		atom_idx = int(atom_idx_str) - 1
		atomic_number = int(atomic_number_str)
		element_id = atomic_number_label_dict.get(atomic_number)
		x, y, z = float(x_str), float(y_str), float(z_str)

	return


def plot_matrices(
		cx: np.array,
		dx: np.array,
		title_info: tuple[str, float, list[int], list[float]] = ("N/A", 0.0, [], [])  # default to tell when to omit
) -> None:
	"""
	Plots connection and distance matrices as colormeshes
	:param cx: connection matrix
	:param dx: distance matrix
	:param title_info: optional additional runtime information, for titling figure
		molecule name, computation duration, cycles in each SCF, energies at each SCF
	:return:
	"""

	max_n_atoms = cx.shape[0]

	xs = np.arange(max_n_atoms)
	ys = np.arange(max_n_atoms)
	xs, ys = np.meshgrid(xs, ys)

	fig, [ax1, ax2] = plt.subplots(1, 2)
	ax1.pcolormesh(xs, ys, cx, cmap="inferno")
	ax1.set_title("Connection matrix")
	ax2.pcolormesh(xs, ys, dx, cmap="inferno")
	ax2.set_title("Distance matrix")

	mol_name, duration, cycles, energies = title_info
	if mol_name != "N/A":
		fig.suptitle(f"{mol_name} (t={duration}s, c={sum(cycles)}, e={energies[-1]}")

	plt.show()


def reindex(xyz: list[str]):
	"""
	Gets a hashmap corresponding to the reindexing order to have the first block be all carbons, the second all
		halogens, and the third all protons
		The key of the hashmap is the initial gaussian index (int formatted as str); the value is the ordered index
			(starting from 0)
	:param xyz:
	:return:
	"""
	gaussian_index_atom_id_hashmap: dict[str: str] = dict()
	gaussian_idx = 1
	for line in xyz:

		atom_id, (x, y, z) = parse_gjf_vector(line)
		gaussian_index_atom_id_hashmap.update({str(gaussian_idx): atom_id})

	atoms = list(gaussian_index_atom_id_hashmap.values())
	n_atoms = len(atoms)

	n_c: int = atoms.count("C")
	n_h: int = atoms.count("H")
	n_x: int = n_atoms - n_c - n_h

	adj_c_idxs = list(range(n_c))
	adj_x_idxs = list(range(n_c, n_c + n_x))
	adj_h_idxs = list(range(n_c + n_x, n_atoms))

	return


def full_workflow():

	max_n_atoms = 30

	gjfs, logs, mol_names = get_job_dirs()
	for idx, gjf_path in enumerate(gjfs):
		# todo reindex the gjf to have all carbons followed by halogens followed by protons

		# get other associated files
		log_path = logs[idx]
		mol_name = mol_names[idx]

		cx, dx = build_matrices(str(gjf_path), max_n_atoms)
		energies, cycles, duration = parse_log(log_path)


if __name__ == '__main__':

	full_workflow()
