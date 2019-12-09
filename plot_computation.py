#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.
import configparser as cp
import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    output_dir = config["DEFAULT"].get("output_directory")

    results = [np.load(os.path.join(output_dir, "chain_{i:02d}.npy".format(i=i)), allow_pickle=True) for i in range(25)]
    beta = [r[()][0]["beta"] for r in results]
    omp_brm_time = 1e-9 * np.array([np.mean([t["omp_brm_time"] for t in r[()]]) for r in results])
    omp_td_time = 1e-9 * np.array([np.mean([t["omp_td_time"] for t in r[()]]) for r in results])
    lars_td_time = 1e-9 * np.array([np.mean([t["lars_td_time"] for t in r[()]]) for r in results])

    plt.figure()
    plt.plot(beta, omp_brm_time, marker="s", label="OMP-BRM")
    plt.plot(beta, omp_td_time, marker="s", label="OMP-TD")
    plt.plot(beta, lars_td_time, marker="s", label="LARS-TD")
    plt.xscale("log")
    plt.legend()
    plt.xlim(max(beta), min(beta))
    plt.xlabel("beta")
    plt.ylabel("CPU Time (s)")
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "computation_time.png"))
