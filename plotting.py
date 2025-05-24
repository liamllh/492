import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from training import load_training_data
from hashmaps import PAHLiteral


def data_distribution():
    """
    Rough estimation of whether a model can differentiate between different types of molecules in training vs. testing
    Also, plots mean log kow values in 3d against the experimental variables (PAH base, halogen ID, # halogens)
    """

    data_compression_factor = 3.4842286109924316
    current_best_error = 0.00211 * data_compression_factor
    current_best_test_error = 0.1520 * data_compression_factor

    pah_map: dict[PAHLiteral: int] = {
        "anthracene": 0,
        "benzaanthracene": 1,
        "pyrene": 2,
        "phenanthrene": 3,
        "triphenyl": 4
    }
    pah_map_inverse: dict[int: PAHLiteral] = {v: k for k, v in pah_map.items()}

    bx, cx, dx, log_kow, names = load_training_data()
    # init array:
    # dim0 = pah type (5)
    # dim1 = n_subs (3)
    # dim2 = bromo or chloro (2)

    arr_out: list[dict[str: Any]] = list()

    for idx, name in enumerate(names):
        subs, halo, pah = name.split("_")
        n_subs = len(subs.split("-"))

        sidx = n_subs - 1
        hidx = 0 if halo == "chloro" else 1
        pidx = pah_map.get(pah)
        df_entry = {"name": name, "base": pah, "substituent": halo, "n_subs": n_subs, "kow": log_kow[idx].item()}
        arr_out.append(df_entry)

    df_out = pd.DataFrame(arr_out)
    df_out.to_csv("./logs/data_table.csv")
    data_arr = np.zeros((5, 3, 2), dtype=np.float32)

    # summaries of each subgroup
    for idx, pah in enumerate(pah_map.keys()):
        for n_subs in range(1, 4):
            for kdx, halo in enumerate(("bromo", "chloro")):
                vals = df_out[(df_out["base"] == pah) & (df_out["substituent"] == halo) & (df_out["n_subs"] == n_subs)]
                mu = np.mean(vals["kow"])
                sigma = np.std(vals["kow"])
                better = current_best_error < sigma
                distance = round(100 * sigma / current_best_error - 1 if not better else 100 * sigma / current_best_error)
                summary = (
                    f"{n_subs}x {halo} {pah}\n"
                    f"n = {vals.shape[0]} | mean = {round(mu.item(), 3)} | std = {round(sigma.item(), 3)} (model {'can' if better else 'cannot'} differentiate [{distance}%])"
                )

                data_arr[idx, n_subs - 1, kdx] = mu

                print(summary)

    print(f"{data_arr = }")

    for pah in pah_map.keys():
        for n_subs in range(1, 4):
            vals = df_out[(df_out["base"] == pah) & (df_out["n_subs"] == n_subs)]
            mu = np.mean(vals["kow"])
            sigma = np.std(vals["kow"])
            better = current_best_error < sigma
            distance = round(100 * sigma / current_best_error - 1 if not better else 100 * sigma / current_best_error)
            summary = (
                f"{n_subs}x {pah}\n"
                f"n = {vals.shape[0]} | mean = {round(mu.item(), 3)} | std = {round(sigma.item(), 3)} (model {'can' if better else 'cannot'} differentiate [{distance}%])"
                )
            print(summary)

    for pah in pah_map.keys():
        vals = df_out[(df_out["base"] == pah)]
        mu = np.mean(vals["kow"])
        sigma = np.std(vals["kow"])
        better = current_best_error < sigma
        distance = round(100 * sigma / current_best_error - 1 if not better else 100 * sigma / current_best_error)
        summary = (
                f"{pah} derivtives\n"
                f"n = {vals.shape[0]} | mean = {round(mu.item(), 3)} | std = {round(sigma.item(), 3)} (model {'can' if better else 'cannot'} differentiate [{distance}%])"
                )
        print(summary)

    mu = np.mean(df_out["kow"])
    sigma = np.std(df_out["kow"])
    better = current_best_error < sigma
    distance = round(100 * sigma / current_best_error - 1 if not better else 100 * sigma / current_best_error)
    summary = (
        f"all derivtives\n"
        f"n = {df_out.shape[0]} | mean = {round(mu.item(), 3)} | std = {round(sigma.item(), 3)} (model {'can' if better else 'cannot'} differentiate [{distance}%])"
    )
    print(summary)

    real_error = 10**current_best_error
    real_test_error = 10**current_best_test_error
    print(
        f"accuracy:\n"
        f"\t[train] log kow best model: +/- {round(current_best_error, 3)} | kow(est) ∈ [{round(100 / real_error, 1)}%, {round(100 * real_error, 1)}%] * kow(expt)\n"
        f"\t[test ] log kow best model: +/- {round(current_best_test_error, 3)} | kow(est) ∈ [{round(100 / real_test_error, 1)}%, {round(100 * real_test_error, 1)}%] * kow(expt)"
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x, y, z = np.arange(5), np.arange(1, 4), np.arange(2)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx = np.transpose(xx, (1, 0, 2))
    yy = np.transpose(yy, (1, 0, 2))
    zz = np.transpose(zz, (1, 0, 2))
    scat = ax.scatter(xx, yy, zz, s=100, c=data_arr, cmap="plasma")
    ax.set_xticks(x, ("A", "BaA", "Py", "PhA", "Ph$_{3}$"))
    ax.set_yticks(y, ("mono", "di", "tri"))
    ax.set_zticks(z, ("bromo", "chloro"))
    cbar = fig.colorbar(scat)
    cbar.set_label("log(k$_{ow}$)")
    plt.show()

    return


if __name__ == '__main__':

    data_distribution()
