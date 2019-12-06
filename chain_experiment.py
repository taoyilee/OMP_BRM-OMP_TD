import warnings

from trial import trial

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    gamma = 0.8
    n_states = 50
    beta = 1e-4

    results = [trial(beta, seed) for seed in range(1000)]
