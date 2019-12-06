import numpy as np

from problems import ChainMRP
from solvers import OMP_BRM, OMP_TD


def trial(beta, seed, n_states=50, gamma=0.8):
    phi = []
    phi_prime = []
    reward = []
    for initial_state in range(1, 51):
        phi_i, phi_prime_i, r_i = ChainMRP(initial_state, n_states, gamma, randomized_seed=seed).step()
        phi.append(phi_i)
        phi_prime.append(phi_prime_i)
        reward.append(r_i)

    initial_state = 1
    chain_problem = ChainMRP(initial_state, n_states, gamma, randomized_seed=seed)
    print(f"State transition matrix P_ss:\n{chain_problem._transition_matrix}")
    print(f"Reward R:\n{chain_problem._reward}")
    print(f"Discount Factor gamma:\n{chain_problem.gamma}")
    Vstar = chain_problem.optimum_value_function
    print(f"Optimum value function V*:\n{Vstar}")
    for _ in range(450):
        phi_i, phi_prime_i, r_i = chain_problem.step()
        phi.append(phi_i)
        phi_prime.append(phi_prime_i)
        reward.append(r_i)
    phi = np.array(phi)
    phi_prime = np.array(phi_prime)
    reward = np.array(reward)
    print(phi.shape, phi_prime.shape, reward.shape)
    w_brm, non_zero, selected_indexes = OMP_BRM(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                               reward=reward)
    vbeta_brm = np.matmul(phi[:50], w_brm)
    error_brm = np.sqrt(np.mean((Vstar - vbeta_brm) ** 2))
    print(f"OMB BRM estimated V*:\n{non_zero} {selected_indexes + 1} {np.sqrt(np.mean((Vstar - vbeta_brm) ** 2)):.2f}")
    w_td, non_zero, selected_indexes = OMP_TD(beta=beta).fit(gamma=gamma, phi=phi, phi_prime=phi_prime,
                                                             reward=reward)
    vbeta_td = np.matmul(phi[:50], w_td)
    error_td = np.sqrt(np.mean((Vstar - vbeta_td) ** 2))
    print(f"OMB TD estimated V*:\n{non_zero} {selected_indexes + 1} {error_td:.2f}")
    return {"beta": beta, "seed": seed, "error_td": error_td, "error_brm": error_brm}
