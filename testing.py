from training import ChemLSTM, load_pretrained_model, pad_square_tensor_zeros
from chemical_types import PAHLiteral
from typing import get_args
from make_files import GJF, randomize_matrices, compute_log_kow
import torch
from torch import Tensor
from pathlib import Path
from directories import sep
from tqdm import tqdm


def decode_output(output: float | Tensor) -> float | Tensor:
    # reverse minmax scale from the model's output of [0, 1] to log kow
    return output * (9.483287811279297 - 5.999059200286865) + 5.999059200286865


def get_testing_data(
        n_repeats: int = 1
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[float], list[str]]:
    """
    Modified version of make_files.get_finished_data
    Focus was on getting this written quickly, not performant
    """

    all_bx_matrices: list[Tensor] = list()
    all_cx_matrices: list[Tensor] = list()
    all_dx_matrices: list[Tensor] = list()
    all_kows: list[float] = list()
    hashmap_key: list[str] = list()

    for pah in get_args(PAHLiteral):
        # get all finished job log files for each PAH base
        finished_job_dir = f".{sep}pah_bases{sep}{pah}_finished{sep}"
        water_jobs = [str(p) for p in Path.iterdir(Path(finished_job_dir)) if "_water.log" in str(p)]

        for water_log in tqdm(water_jobs, desc=f"Generating test inputs for {pah}"):
            # reformat "a-b-c_halo_pah_water.log" as "a-b-c_halo_pah"
            substitutional_pattern_str, halo_prefix, base = water_log.split(sep)[-1].split("_")[:-1]
            xpah_id = f"{substitutional_pattern_str}_{halo_prefix}_{base}"
            # get the octanol job by substitution in the water log filename
            octanol_log = water_log[:-9] + "octanol.log"
            # get the optimized structure
            base_gjf_dir = f".{sep}pah_bases{sep}{pah}.gjf"
            base_structure = GJF(base_gjf_dir)

            # screen for data outliers - excluded values outside of [5, 10]
            # 3 outliers were found in the dataset of 284, and I intend to rerun these jobs assuming the values
            # represent a local SCF minimum rather than physical properties (all were tribromo triphenyl derivatives)
            # outliers:
            #   1-5-8 tribromo triphenyl: log kow = 13.88
            #   0-4-8 tribromo triphenyl: log kow = 38.71
            #   1-4-9 tribromo triphenyl: log kow = 3.37
            log_kow = compute_log_kow(water_log, octanol_log)
            if log_kow > 10:
                print(f"abnormal log kow detected: {log_kow} @ {xpah_id} | skipping")
                continue
            if log_kow < 5:
                print(f"abnormal log kow detected: {log_kow} @ {xpah_id} | skipping")
                continue

            # passed the outlier screen, so save this molecule in the key
            hashmap_key.append(xpah_id)

            # because we extract the distance matrix from the unoptimized structure, these distances will not be the
            # lowest energy conformer in the respective solvent. However, the model should be powered entirely by
            # connective data, so the mismatch should not matter
            # distance matrix
            base_dx = base_structure.get_normalized_distance_matrix()
            # reduced mass matrix of atoms at row-column intersections; -(reduced mass) if self-intersection
            base_cx = base_structure.get_connection_matrix()
            # bond order matrix; -1 if self-intersection
            base_bx = base_structure.get_bond_order_matrix()

            # generate n_repeats redundant representations of the same molecules
            # we must pass in (bx, cx, dx) at the same time so they are randomized by the same key to retain parity
            base_bxs, base_cxs, base_dxs = randomize_matrices(base_bx, base_cx, base_dx, n_repeats)
            base_bxs = [pad_square_tensor_zeros(b, 30) for b in base_bxs]
            base_cxs = [pad_square_tensor_zeros(c, 30) for c in base_cxs]
            base_dxs = [pad_square_tensor_zeros(d, 30) for d in base_dxs]
            # stack to batch
            all_bx_matrices.append(torch.stack(base_bxs, dim=0))
            all_cx_matrices.append(torch.stack(base_cxs, dim=0))
            all_dx_matrices.append(torch.stack(base_dxs, dim=0))
            all_kows.append(log_kow)

            # print(f"{water_log.split(sep)[-1].replace('_water.log', '')} | log(kow) = {log_kow}")

    return all_bx_matrices, all_cx_matrices, all_dx_matrices, all_kows, hashmap_key


def test_model(pretrained_model_name: str, n_redundant: int = 2**10):
    """
    Tests the model at directory pretrained_model_name on
    """
    # load model
    model, _, __, ___, ____, _____ = load_pretrained_model(pretrained_model_name)
    # set testing mode
    model.train(False)

    # load pah base info
    pah_base_paths = ([f"./pah_bases/{k}.gjf" for k in get_args(PAHLiteral)])
    pah_models = [GJF(path) for path in pah_base_paths]
    pah_dx_matrices = torch.stack([pad_square_tensor_zeros(gjf.get_normalized_distance_matrix().to(torch.device("cuda:0")), 30) for gjf in pah_models])
    pah_cx_matrices = torch.stack([torch.tanh(pad_square_tensor_zeros(gjf.get_connection_matrix().to(torch.device("cuda:0")) / 12, 30)) for gjf in pah_models], dim=0)
    pah_bx_matrices = torch.stack([pad_square_tensor_zeros(gjf.get_bond_order_matrix().to(torch.device("cuda:0")), 30) for gjf in pah_models], dim=0)

    # create redundant copies of pah base info for testing
    rx = [randomize_matrices(pah_bx_matrices[i], pah_cx_matrices[i], pah_dx_matrices[i], n_redundant) for i in range(len(pah_models))]

    # run test on redundant copies and get result statistics
    for i in range(len(pah_base_paths)):
        randomized_bx, randomized_cx = rx[i][:-1]
        bx = torch.stack(randomized_bx, dim=0)
        cx = torch.stack(randomized_cx, dim=0)
        inputs = torch.concat((bx, cx), dim=-1)
        results: list[float] = list()

        for j in range(n_redundant):
            out = model.forward(inputs[j, :, :])
            results.append(decode_output(out[0]))

        results_tensor = Tensor(results)
        print(f"{pah_base_paths[i]} | mean = {torch.mean(results_tensor).item()} | std = {torch.std(results_tensor).item()}")

    # do the same for each sample in run derivatives
    all_bx_matrices, all_cx_matrices, all_dx_matrices, all_kows, hashmap_key = get_testing_data(n_redundant)
    n_run = len(all_bx_matrices)
    for i in range(n_run):
        bx = all_bx_matrices[i]
        cx = all_cx_matrices[i]
        inputs = torch.concat((bx, cx), dim=-1).to(model.device)
        results: list[float] = list()
        for j in range(n_redundant):
            out = model.forward(inputs[j, :, :])
            results.append(decode_output(out[0]))

        results_tensor = Tensor(results)
        print(f"{hashmap_key[i]} | mean = {torch.mean(results_tensor).item()} "
              f"| std = {torch.std(results_tensor).item()} "
              f"| mean err = {torch.mean(all_kows[i] - results_tensor).item()} "
              f"| RMSE = {torch.mean((all_kows[i] - results_tensor)**2).item()**0.5}")


if __name__ == '__main__':
    test_model("expt_2")
