#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.


import numpy as np

from problems import ChainMRP5
from solvers import OMP_BRM, OMP_TD, LARS_TD

if __name__ == "__main__":
    gamma = 0.8
    chain_problem = ChainMRP5(gamma)
    beta = 1e-9
    print(chain_problem.n_states)
    print(f"State transition matrix P_ss:\n{chain_problem.transition_matrix}")
    print(f"Reward R:\n{chain_problem.reward}")
    print(f"Discount Factor gamma:\n{chain_problem.gamma}")
    print(f"Optimum value function V*:\n{chain_problem.optimum_value_function}")
    phi = np.eye(5)
    phi_prime = np.matmul(chain_problem.transition_matrix, phi)
    coeff, non_zeros, selected_indexes = OMP_BRM(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                                reward=chain_problem.reward)

    print(f"OMP BRM estimated V*:\n{coeff} {non_zeros} {selected_indexes}")
    coeff_td, non_zeros, selected_indexes = OMP_TD(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                                  reward=chain_problem.reward)
    print(f"OMP TD estimated V*:\n{coeff_td} {non_zeros} {selected_indexes}")

    coeff_td, non_zeros, selected_indexes = LARS_TD(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                                   reward=chain_problem.reward)
    print(f"LARS TD estimated V*:\n{coeff_td} {non_zeros} {selected_indexes}")
