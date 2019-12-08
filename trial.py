#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.


import time

import numpy as np

from problems import ChainMRP
from solvers import OMP_BRM, OMP_TD, LARS_TD


def trial(beta, seed, n_states=50, gamma=0.8, samples=500):
    phi = []
    phi_prime = []
    reward = []
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
    phi_50 = np.array([c.rbf.phi(s) for s in range(1, n_states + 1)])
    start = time.perf_counter_ns()
    w_brm, non_zero_brm, selected_indexes = OMP_BRM(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                                   reward=reward)
    end = time.perf_counter_ns()
    omp_brm_time = end - start

    vbeta_brm = np.matmul(phi_50, w_brm)
    error_brm = np.sqrt(np.mean((vstar - vbeta_brm) ** 2))

    start = time.perf_counter_ns()
    w_td, non_zero_td, selected_indexes = OMP_TD(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                                reward=reward)
    end = time.perf_counter_ns()
    omp_td_time = end - start

    vbeta_td = np.matmul(phi_50, w_td)
    error_td = np.sqrt(np.mean((vstar - vbeta_td) ** 2))
    # lars -TD
    start = time.perf_counter_ns()
    w_lars_td, non_zero_lars_td, selected_indexes = LARS_TD(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                                           reward=reward)
    end = time.perf_counter_ns()
    lars_td_time = end - start

    vbeta_laras_td = np.matmul(phi_50, w_lars_td)
    error_lars_td = np.sqrt(np.mean((vstar - vbeta_laras_td) ** 2))
    print(non_zero_brm, non_zero_td, non_zero_lars_td)
    print(f"{omp_brm_time:.2f}, {omp_td_time:.2f}, {lars_td_time:.2f}")
    return {"beta": beta, "seed": seed,
            "error_td": error_td,
            "error_brm": error_brm,
            "error_lars_td": error_lars_td,
            "vstar": vstar,
            "reward": c._reward,
            "omp_brm_time": omp_brm_time,
            "omp_td_time": omp_td_time,
            "lars_td_time": lars_td_time,
            "non_zero_brm": non_zero_brm,
            "non_zero_td": non_zero_td,
            "non_zero_lars_td": non_zero_lars_td,
            "state_transition": c._transition_matrix}
