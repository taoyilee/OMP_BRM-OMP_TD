import os
import warnings

import numpy as np

from trial import trial

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    output_dir = "/home/tylee/michael@fraxense.ai/UCI/05_Fall_2019/CS295_RL/project/plot"
    gamma = 0.8
    n_states = 50
    mean_error_td = []
    mean_error_brm = []
    std_error_td = []
    std_error_brm = []
    beta = np.logspace(-1, -7, 10)
    for b in beta:
        print(f"beta = {b}")
        results = [trial(b, seed) for seed in range(10)]
        mean_error_td.append(np.mean([r["error_td"] for r in results]))
        mean_error_brm.append(np.mean([r["error_brm"] for r in results]))
        std_error_td.append(np.std([r["error_td"] for r in results]))
        std_error_brm.append(np.std([r["error_brm"] for r in results]))

    plt.errorbar(beta, mean_error_brm, yerr=std_error_brm, color="C0", fmt='-s', label="OMP-BRM")
    plt.errorbar(beta, mean_error_td, yerr=std_error_td, color="C1", fmt='-s', label="OMP-TD")
    plt.legend()
    plt.grid()
    plt.xscale("log")
    plt.xlim(10 ** -1, 10 ** -7)
    plt.xlabel("beta")
    plt.ylabel("||V* - Vp||")
    plt.savefig(os.path.join(output_dir, "chain.png"))
