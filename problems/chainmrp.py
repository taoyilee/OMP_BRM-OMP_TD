import numpy as np


def gaussian_rbf(x, shaping_factor, center):
    r = np.abs(x - center)
    return np.exp(-(shaping_factor * r) ** 2)


class RBF50:
    def __init__(self, states=50, n_basis=208, shaping_factor=1, random=False, seed=0, rbf_constant=1):
        self.n_basis = n_basis
        np.random.seed(seed)
        self.rbf_constant = rbf_constant
        if random:
            self.shaping_factor = 10 * np.random.random(n_basis)
            self.center = (states + 1) * np.random.random(n_basis)
        else:
            self.shaping_factor = shaping_factor * np.ones(n_basis - 1)
            self.center = np.linspace(0, states + 1, n_basis - 1)

    def phi(self, state):
        return np.array([self.rbf_constant] +
                        [gaussian_rbf(state, self.shaping_factor[0], self.center[i]) for i in range(self.n_basis - 1)])


class ChainMRP:
    TRANSITION_RIGHT = 1
    TRANSITION_LEFT = 1
    TRANSITION_STAY = 1
    _p = None
    _r = None
    _current_state: int = None

    @property
    def _next_state(self):
        return int(np.random.choice(np.arange(1, self.n_states + 1), size=None,
                                    p=self._transition_matrix[self._current_state - 1, :]))

    @property
    def phi(self):
        return self.rbf.phi(self._current_state)

    @property
    def current_reward(self):
        return int(self._reward[self._current_state - 1])

    def step(self):
        phi = self.phi
        new_state = self._next_state
        self._current_state = new_state
        return phi, self.phi, self.current_reward

    def __init__(self, initial_state: int, number_of_states=50, gamma=0.8, is_noisy=False, min_reward=0, max_reward=2,
                 randomized_seed=None):
        self._n = number_of_states
        self._gamma = gamma
        self.is_noisy = is_noisy
        self.rbf = RBF50(self.n_states)
        if initial_state not in list(range(1, number_of_states + 1)):
            raise ValueError(f"initial_state much in 1 .. {number_of_states}")

        self._current_state = initial_state
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.randomized_seed = randomized_seed
        if self.randomized_seed is not None:
            np.random.seed(self.randomized_seed)
        assert isinstance(self._transition_matrix, np.ndarray)
        assert isinstance(self._reward, np.ndarray)

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
    def _transition_matrix(self) -> np.ndarray:
        """
        State transition matrix P[s_t+1 | s_t] of the MRP
        Dimension: n_states x n_states
        :return:
        """
        if self._p is None:
            self._p = np.zeros((self.n_states, self.n_states))
            for i in range(0, self.n_states):
                if self.randomized_seed is not None:
                    self._p[i, i] = np.random.rand()
                    self._p[i, max(0, i - 1)] = np.random.rand()
                    self._p[i, min(self.n_states - 1, i + 1)] = np.random.rand()
                else:
                    self._p[i, i] = self.TRANSITION_STAY
                    self._p[i, max(0, i - 1)] = self.TRANSITION_LEFT
                    self._p[i, min(self.n_states - 1, i + 1)] = self.TRANSITION_RIGHT
            self._p /= self._p.sum(axis=1)[:, np.newaxis]  # normalize
        return self._p

    @property
    def _reward(self) -> np.ndarray:
        """
        Reward vector, randomly generated from 0 to 10
        Dimension: n_states x 1
        :return:
        """
        if self._r is None:
            self._r = np.random.randint(self.min_reward, self.max_reward, self.n_states)
        return self._r

    @property
    def optimum_value_function(self):
        return np.matmul(np.linalg.inv(np.eye(self.n_states) - self.gamma * self._transition_matrix), self._reward)
