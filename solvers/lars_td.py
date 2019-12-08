#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.


import numpy as np

from solvers.helper import SelectedIndexes


def min_plus(x):
    x = np.where(x > 0, x, np.inf)
    j = x.argmin()
    return x[j], j


def max_abs(x):
    x = np.abs(x)
    j = x.argmax()
    return x[j], j


class LARS_TD:
    """
    Reference:
    Kolter, J. Zico, and Andrew Y. Ng. "Regularization and feature selection in least-squares temporal difference
     learning." Proceedings of the 26th annual international conference on machine learning. ACM, 2009.
    """

    def __init__(self, beta=np.finfo(float).eps):
        self.beta = beta

    def fit(self, gamma, phi: "np.ndarray", phi_prime: "np.ndarray", reward):
        n, k = phi.shape
        w = np.zeros((k, 1))
        reward = reward.reshape(len(reward), 1)
        I = SelectedIndexes(k)
        # initialization
        c = np.matmul(phi.T, reward)
        beta_bar, i = max_abs(c)  # i.e., residue
        I.add(i)

        while len(I) < k and beta_bar > n * self.beta:
            phi_i = phi[:, I.I]
            phi_i_prime = phi_prime[:, I.I]
            aii = np.matmul(phi_i.T, phi_i - gamma * phi_i_prime)
            delta_w = np.matmul(np.linalg.inv(aii), np.where(c[I.I] >= 0, 1, -1))
            d = np.matmul(np.matmul(phi.T, phi_i - gamma * phi_i_prime), delta_w)
            left_term, left_i = min_plus((beta_bar - c[I.IBar]) / (d[I.IBar] - 1))
            right_term, right_i = min_plus((beta_bar + c[I.IBar]) / (d[I.IBar] + 1))
            alpha1, i1 = (left_term, left_i) if left_term < right_term else (right_term, right_i)
            i1 = I.IBar[i1]

            alpha2, i2 = min_plus(-w[I.I] / delta_w)
            i2 = I.I[i2]
            alpha = np.min([alpha1, alpha2, beta_bar - n * self.beta])
            w[I.I], beta_bar, c = w[I.I] + alpha * delta_w, beta_bar - alpha, c - alpha * d

            if alpha1 < alpha2:
                print(f"Add {i1}")
                I.add(i1)
            else:
                print(f"Remove {i2}")
                I.remove(i2)

        return w.squeeze(), len(I), np.array(I.I)
