#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.


import warnings

import numpy as np

from problems import ChainMRP
from solvers import LARS_TD

warnings.filterwarnings('ignore')
if __name__ == "__main__":
    seed = 0
    gamma = 0.8
    n_states = 50
    mean_error_td = []
    mean_error_brm = []
    std_error_td = []
    std_error_brm = []
    higher_exp = -1
    lower_exp = -7
    beta = 1e-3

    print(f"beta = {beta}")
    phi = []
    phi_prime = []
    reward = []
    samples = 500
    c = ChainMRP(1, n_states, gamma, randomized_seed=seed)
    vstar = c.optimum_value_function
    for s in range(1, n_states + 1):
        c._current_state = s
        for _ in range(samples // n_states):
            phi_i, phi_prime_i, r_i = c.step()
            phi.append(phi_i)
            phi_prime.append(phi_prime_i)
            reward.append(r_i)

    phi = np.array(phi)
    phi_prime = np.array(phi_prime)
    reward = np.array(reward)
    print(phi.shape, phi_prime.shape, reward.shape)
    phi_50 = np.array([c.rbf.phi(s) for s in range(1, n_states + 1)])

    w_td, non_zero, selected_indexes = LARS_TD(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                              reward=reward)
    print(w_td)
    print(selected_indexes)
    print(vstar)
    print(np.matmul(phi_50, w_td))
    print(np.round(np.abs(vstar - np.matmul(phi_50, w_td)), 2))
