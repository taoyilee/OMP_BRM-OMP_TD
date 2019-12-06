import numpy as np

from solvers.helper import SelectedIndexes


class OMP_TD:
    def __init__(self, beta=np.finfo(float).eps):
        self.beta = beta

    def fit(self, gamma, phi: "np.ndarray", phi_prime: "np.ndarray", reward):
        residue = None
        n, k = phi.shape
        w = np.zeros((k, 1))
        reward = reward.reshape(len(reward), 1)
        I = SelectedIndexes(k)

        while len(I) < k and (residue is None or residue > self.beta):
            c = np.linalg.norm(np.matmul(phi.T, reward + gamma * np.matmul(phi_prime, w) - np.matmul(phi, w)),
                               axis=1, ord=1) / n
            j = I.IBar[int(np.argmax(c[I.IBar]))]
            residue = c[j]
            if residue > self.beta:
                I.add(j)
            phi_i = phi[:, I.I]
            phi_i_prime = phi_prime[:, I.I]
            w[I.I] = np.matmul(
                np.matmul(np.linalg.inv(np.matmul(phi_i.T, phi_i) - gamma * np.matmul(phi_i.T, phi_i_prime)), phi_i.T),
                reward)
        return w.squeeze(), len(I), np.array(I.I)

