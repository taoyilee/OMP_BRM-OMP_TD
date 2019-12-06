import numpy as np

from experiments import OMP_BRM, OMP_TD
from experiments.problems import ChainMRP

if __name__ == "__main__":
    gamma = 0.8
    n_states = 50
    initial_state = 1
    chain_problem = ChainMRP(initial_state, n_states, gamma, randomized_seed=0)
    beta = 0.1
    omp_brm = OMP_BRM(beta=beta)
    print(chain_problem.n_states)
    print(f"State transition matrix P_ss:\n{chain_problem._transition_matrix}")
    print(f"Reward R:\n{chain_problem._reward}")
    print(f"Discount Factor gamma:\n{chain_problem.gamma}")
    print(f"Optimum value function V*:\n{chain_problem.optimum_value_function}")

    phi = np.eye(n_states)
    phi_prime = np.matmul(chain_problem._transition_matrix, phi)
    coeff, non_zero, selected_indexes = omp_brm.fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                    reward=chain_problem._reward)
    print(f"OMB BRM estimated V*:\n{coeff.T} {non_zero} {selected_indexes + 1}")

    omp_td = OMP_TD(beta=beta)
    coeff_td, non_zero, selected_indexes = omp_td.fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                      reward=chain_problem._reward)
    print(f"OMB TD estimated V*:\n{coeff_td.T} {non_zero} {selected_indexes + 1}")
