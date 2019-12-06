import numpy as np

from experiments import OMP_BRM, OMP_TD
from experiments.problems import ChainMRP5

if __name__ == "__main__":
    gamma = 0.8
    chain_problem = ChainMRP5(gamma)
    omp_brm = OMP_BRM()
    print(chain_problem.n_states)
    print(f"State transition matrix P_ss:\n{chain_problem.transition_matrix}")
    print(f"Reward R:\n{chain_problem.reward}")
    print(f"Discount Factor gamma:\n{chain_problem.gamma}")
    print(f"Optimum value function V*:\n{chain_problem.optimum_value_function}")
    phi = np.eye(5)
    phi_prime = np.matmul(chain_problem.transition_matrix, phi)
    coeff = omp_brm.fit(gamma=gamma, phi=phi, phi_prime=phi_prime, reward=chain_problem.reward)

    print(f"OMB BRM estimated V*:\n{coeff.T}")

    omp_td = OMP_TD()
    coeff_td, selected_indexes = omp_td.fit(gamma=gamma, phi=phi, phi_prime=phi_prime, reward=chain_problem.reward)
    print(f"OMB TD estimated V*:\n{coeff_td.T} {selected_indexes}")
