import numpy as np


class SelectedIndexes:
    _I = None
    _I_bar = None

    def __init__(self, n: int):
        self.n = n
        self._I = []

    def add(self, i):
        self._I.append(i)
        self._I.sort()
        self._I_bar.remove(i)

    @property
    def I(self) -> list:
        return self._I

    @property
    def IBar(self) -> list:
        if self._I_bar is None:
            self._I_bar = list(set(list(range(self.n))) - set(self.I))
            self._I_bar.sort()
        return self._I_bar

    def __len__(self):
        return len(self._I)


class OMP_TD:
    def __init__(self, beta=np.finfo(float).eps):
        self.beta = beta

    def fit(self, gamma, phi: "np.ndarray", phi_prime: "np.ndarray", reward):
        residue = None
        n, k = phi.shape
        w = np.zeros((k, 1))
        reward = reward.reshape(len(reward), 1)
        I = SelectedIndexes(n)

        while len(I) < n and (residue is None or residue > self.beta):
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

