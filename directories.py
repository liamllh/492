# todo switch every string path to Path path, this is ugly
system = "unix"
if system == "unix":
    sep = "/"
else:
    sep = "\\"

base_gjf_path: str = f".{sep}pah_bases{sep}"

anthracene_derivatives_path = base_gjf_path + f"anthracene_derivatives{sep}"
phenanthrene_derivatives_path = base_gjf_path + f"phenanthrene_derivatives{sep}"
pyrene_derivatives_path = base_gjf_path + f"pyrene_derivatives{sep}"
benzaanthracene_derivatives_path = base_gjf_path + f"benzaanthracene_derivatives{sep}"
triphenyl_derivatives_path = base_gjf_path + f"triphenyl_derivatives{sep}"

derivatives_paths: list[str] = [anthracene_derivatives_path, phenanthrene_derivatives_path, pyrene_derivatives_path, benzaanthracene_derivatives_path, triphenyl_derivatives_path]
derivatives_paths_run: list[str] = [path[:-1] + "_run" + sep for path in derivatives_paths]
