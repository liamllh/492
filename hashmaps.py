from chemical_types import PAHLiteral
from congruency_tables import *
from directories import *


# number of samples for each pah type (keys) for each number of halogen substituents (values)
sample_dict: dict[PAHLiteral: tuple[int]] = {
	"anthracene": (3, 5, 13),
	"phenanthrene": (5, 8, 15),
	"pyrene": (3, 7, 14),
	"triphenyl": (2, 6, 17),
	"benzaanthracene": (4, 15, 25)
}

# order of atom blocks for reindexing gjf files; C must be 0
# currently unused, went with a different implementation for sorting
atom_priority_dict: dict[str: int] = {
	"C": 0,
	"H": 1,
	"Cl": 2,
	"Br": 3,
}

atomic_label_weight_dict: dict[str: float] = {
	"C": 12.011,
	"H": 1.008,
	"Cl": 35.453,
	"Br": 79.904,
}

# atomic numbers of atoms used in this project
atomic_number_label_dict: dict[int: str] = {
	1: "H",
	12: "C",
	17: "Cl",
	35: "Br",
}
# inverse of above
atomic_label_number_dict: dict[int: str] = {
	"H": 1,
	"C": 12,
	"Cl": 17,
	"Br": 35,
}

# for coloring plots, see make_files.GJF.plot()
atomic_label_rgba_dict: dict[str: tuple[float, float, float, float]] = {
	"H": (0, 0, 0, 0.2),  # light grey
	"C": (0, 0, 0, 0.6),  # dark grey
	"Cl": (0.8, 1, 0, 1),  # yellow
	"Br": (1, 0.6, 0, 1)  # orange
}

# links each base with its symmetries
congruency_dict: dict[PAHLiteral: list[list[list[int]]]] = {
	"anthracene": anthracene_congruency,
	"phenanthrene": phenanthrene_congruency,
	"pyrene": pyrene_congruency,
	"triphenyl": triphenyl_congruency,
	"benzaanthracene": benzaanthracene_congruency,
}

# links each base with its unsubstituted gjf filepath
base_pah_filepath_dict: dict[PAHLiteral: str] = {
	"anthracene": base_gjf_path + "anthracene.gjf",
	"phenanthrene": base_gjf_path + "phenanthrene.gjf",
	"pyrene": base_gjf_path + "pyrene.gjf",
	"triphenyl": base_gjf_path + "triphenyl.gjf",
	"benzaanthracene": base_gjf_path + "benzaanthracene.gjf",
}

# correspondence dicts are offset by 1 from the atomic labels in the gjf because we want to index line numbers, not labels
# this matches the indexing in GJF objects (see make_files.GJF)
anthracene_correspondence_dict: dict[int: int] = {
	0: 21, 1: 11, 2: 15,
	3: 17, 4: 16, 5: 19,
	6: 18, 7: 20, 8: 23,
	9: 22
}
pyrene_correspondence_dict: dict[int: int] = {
	0: 24, 1: 25, 2: 18,
	3: 17, 4: 22, 5: 20,
	6: 21, 7: 19, 8: 23,
	9: 11
}
phenanthrene_correspondence_dict: dict[int: int] = {
	0: 21, 1: 22, 2: 23,
	3: 18, 4: 19, 5: 16,
	6: 17, 7: 15, 8: 20,
	9: 11
}
triphenyl_correspondence_dict: dict[int: int] = {
	0: 14, 1: 17, 2: 19,
	3: 18, 4: 4, 5: 7,
	6: 9, 7: 8, 8: 24,
	9: 27, 10: 29, 11: 28,
}
benzaanthracene_correspondence_dict: dict[int: int] = {
	0: 17, 1: 16, 2: 18,
	3: 28, 4: 29, 5: 27,
	6: 24, 7: 19, 8: 11,
	9: 13, 10: 15, 11: 14,
}

# aligns PAH literals with their correspondence hashmaps (pah: correspondence)
correspondence_dict: dict[PAHLiteral: dict[int: int]] = {
	"anthracene": anthracene_correspondence_dict,
	"phenanthrene": phenanthrene_correspondence_dict,
	"pyrene": pyrene_correspondence_dict,
	"triphenyl": triphenyl_correspondence_dict,
	"benzaanthracene": benzaanthracene_correspondence_dict
}

# prefixes for n-substituted molecules (n: prefix)
substitution_prefix_dict: dict[int: str] = {
	1: "", 2: "di", 3: "tri"
}
