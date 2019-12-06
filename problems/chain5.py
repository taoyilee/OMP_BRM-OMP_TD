import numpy as np




class ChainMRP5:
    _p = None
    _n = 5

    def __init__(self, gamma=0.8, is_noisy=False):
        self._gamma = gamma
        self.is_noisy = is_noisy

    @property
    def n_states(self) -> int:
        """
        Number of states
        :return:
        """
        return self._n

    @property
    def gamma(self) -> float:
        """
        Discount factor
        :return:
        """
        return self._gamma

    @property
    def transition_matrix(self) -> np.ndarray:
        """
        State transition matrix P[s_t+1 | s_t] of the MRP
        Dimension: n_states x n_states
        :return:
        """
        if self._p is None:
            self._p = np.zeros((self.n_states, self.n_states))
            for i in range(0, self._p.shape[0] - 1):
                self._p[i, i + 1] = 1
            self._p[-1, -1] = 1
            self._p /= self._p.sum(axis=1)[:, np.newaxis]  # normalize
        return self._p

    @property
    def reward(self) -> np.ndarray:
        """
        Reward vector, randomly generated from 0 to 10
        Dimension: n_states x 1
        :return:
        """
        return np.array([-(self.gamma + self.gamma ** 2 + self.gamma ** 3), 1, 1, 1, 0])

    @property
    def optimum_value_function(self):
        return np.matmul(np.linalg.inv(np.eye(self.n_states) - self.gamma * self.transition_matrix), self.reward)
