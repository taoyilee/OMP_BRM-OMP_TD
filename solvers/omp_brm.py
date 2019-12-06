import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


class OMP_BRM:
    def __init__(self, beta=np.finfo(float).eps):
        self.beta = beta

    def fit(self, gamma, phi, phi_prime, reward):
        omp = OrthogonalMatchingPursuit(tol=self.beta, normalize=False, fit_intercept=False)
        omp.fit(phi - gamma * phi_prime, reward)
        return omp.coef_, omp.n_iter_, np.array(np.nonzero(omp.coef_)[0])
