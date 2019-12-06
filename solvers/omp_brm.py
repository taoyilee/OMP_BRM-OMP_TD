import numpy as np

from solvers.helper import SelectedIndexes
from sklearn.linear_model import OrthogonalMatchingPursuit


class OMP_BRM:
    def __init__(self, beta):
        self.beta = beta

    def fit(self, gamma, phi, phi_prime, reward):
        residue = None
        n, k = phi.shape
        w = np.zeros((k, 1))
        reward = reward.reshape(len(reward), 1)
        I = SelectedIndexes(k)
        X = phi - gamma * phi_prime
        while len(I) < k and (residue is None or residue > self.beta):
            c = np.linalg.norm(np.matmul(X.T, reward - np.matmul(X, w)), axis=1, ord=1) / n
            j = I.IBar[int(np.argmax(c[I.IBar]))]
            residue = c[j]
            if residue > self.beta:
                I.add(j)
            w[I.I] = np.matmul(np.linalg.pinv(X[:, I.I]), reward)
        return w.squeeze(), len(I), np.array(I.I)
