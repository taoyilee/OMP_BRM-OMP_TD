#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.


import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from trial import trial

warnings.filterwarnings('ignore')
if __name__ == "__main__":
    output_dir = "/home/tylee/michael@fraxense.ai/UCI/05_Fall_2019/CS295_RL/project/plot"
    gamma = 0.8
    n_states = 50
    mean_error_td = []
    mean_error_brm = []
    mean_error_lars_td = []
    std_error_td = []
    std_error_brm = []
    std_error_lars_td = []
    higher_exp = -1
    lower_exp = -5
    beta = np.logspace(higher_exp, lower_exp, 5)
    for i, b in enumerate(beta):
        print(f"beta = {b}")
        results = [trial(b, seed) for seed in range(5)]
        mean_error_td.append(np.mean([r["error_td"] for r in results]))
        mean_error_brm.append(np.mean([r["error_brm"] for r in results]))
        mean_error_lars_td.append(np.mean([r["error_lars_td"] for r in results]))
        std_error_td.append(np.std([r["error_td"] for r in results]))
        std_error_brm.append(np.std([r["error_brm"] for r in results]))
        std_error_lars_td.append(np.std([r["error_lars_td"] for r in results]))
        np.save(os.path.join(output_dir, f"chain_{i:02d}.npy"), results)

    plt.errorbar(beta, mean_error_brm, yerr=std_error_brm, color="C0", fmt='-s', label="OMP-BRM")
    plt.errorbar(beta, mean_error_td, yerr=std_error_td, color="C1", fmt='-s', label="OMP-TD")
    plt.errorbar(beta, mean_error_lars_td, yerr=std_error_lars_td, color="C2", fmt='-s', label="LARS-TD")
    plt.xscale("log")
    plt.xlim(10 ** higher_exp, 10 ** lower_exp)
    plt.xlabel("beta")
    plt.ylabel("||V* - Vp||")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chain.png"))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chain.png"))
