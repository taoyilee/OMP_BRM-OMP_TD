#  Copyright (c) 2019.
#  Author:  Tao-Yi Lee
#  This source code is released under MIT license. Check LICENSE for details.
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


def fetch_diabetes(subset='train'):
    diabetes = sklearn.datasets.load_diabetes()
    X_all = diabetes.data
    y_all = diabetes.target

    total_N = len(y_all)
    train_N = int(total_N * diabetes_train_splitsize)
    test_N = total_N - train_N
    rand = np.random.mtrand.RandomState(seed=123)
    train_idx = set(rand.choice(total_N, size=(train_N,), replace=False))

    train_X = X_all[list(train_idx)]
    train_y = y_all[list(train_idx)]

    test_idx = np.zeros((test_N,), dtype=np.int32)
    test_n = 0
    for n in range(total_N):
        if n not in train_idx:
            test_idx[test_n] = n
            test_n += 1
    test_X = X_all[test_idx]
    test_y = y_all[test_idx]

    def get_add_mul(X):
        add = - np.average(X, 0)
        X1 = X + add
        mul = 1 / np.sqrt((X1 * X1).sum(0))
        return add, mul

    X_add, X_mul = get_add_mul(train_X)
    y_add = - np.average(train_y)

    train_X = (train_X + X_add) * X_mul
    train_y = train_y + y_add
    if len(test_X) > 0:
        test_X = (test_X + X_add) * X_mul
        test_y = test_y + y_add

    if subset == 'train':
        return train_X, train_y
    elif subset == 'test':
        return test_X, test_y
    else:
        raise Exception('unknown subset %s' % subset)


def vector_len(vector):
    return np.sqrt(np.sum(vector * vector))


def run_lars(train_data, train_target, tol=np.finfo(float).eps):
    X = train_data
    y = train_target
    m = len(X[0])
    n = len(X)
    active_set = set()
    cur_pred = np.zeros((n,), dtype=np.float32)
    residual = y - cur_pred
    cur_corr = X.transpose().dot(residual)
    j = np.argmax(np.abs(cur_corr), 0)
    active_set.add(j)
    beta = np.zeros((m,), dtype=np.float32)
    sign = np.zeros((m,), dtype=np.int32)
    sign[j] = 1
    beta_path = np.zeros((m, m), dtype=np.float32)
    it = 0

    beta_bar = np.abs(cur_corr)[j]
    print(f"tol = {beta_bar:.3f}")
    while it < m and beta_bar > tol:
        residual = y - cur_pred
        cur_corr = X.T.dot(residual)
        X_a = X[:, list(active_set)]
        X_a *= sign[list(active_set)]
        G_a = X_a.T.dot(X_a)
        G_a_inv = np.linalg.inv(G_a)
        G_a_inv_red_cols = np.sum(G_a_inv, 1)
        A_a = 1 / np.sqrt(np.sum(G_a_inv_red_cols))
        omega = A_a * G_a_inv_red_cols
        equiangular = X_a.dot(omega)

        cos_angle = X.T.dot(equiangular)
        alpha1 = None
        beta_bar = np.abs(cur_corr).max()
        print(np.abs(cur_corr))
        next_j1 = None
        next_sign = 0

        if len(active_set) - 1 < m - 1:
            for j in range(m):
                if j in active_set:
                    continue
                v0 = (beta_bar - cur_corr[j]) / (A_a - cos_angle[j]).item()
                v1 = (beta_bar + cur_corr[j]) / (A_a + cos_angle[j]).item()
                if v0 > 0 and (alpha1 is None or v0 < alpha1):
                    next_j1 = j
                    alpha1 = v0
                    next_sign = +1
                if v1 > 0 and (alpha1 is None or v1 < alpha1):
                    alpha1 = v1
                    next_j1 = j
                    next_sign = -1
        else:
            alpha1 = beta_bar / A_a
        # alpha2 = None
        # next_j2 = None
        # if len(active_set) - 1 < m - 1:
        #     for j in range(m):
        #         if j not in active_set:
        #             continue
        #         v0 = -beta[j] / delta_beta_j
        #         if v0 > 0 and (alpha2 is None or v0 < alpha2):
        #             next_j2 = j
        #             alpha2 = v0

        beta_bar -= alpha1
        print(f"tol = {beta_bar:.3f}")
        estimate = np.matmul(np.linalg.pinv(X_a), equiangular * alpha1)
        beta[list(active_set)] += estimate * sign[list(active_set)]
        cur_pred = X.dot(beta)
        active_set.add(next_j1) if next_j1 is not None else None
        sign[next_j1] = next_sign
        beta_path[len(active_set) - 1, :] = beta
        it += 1
    beta_path = beta_path[:it, :]
    return beta_path


if __name__ == "__main__":
    diabetes_train_splitsize = 1.0
    train_data, train_target = fetch_diabetes(subset='train')
    beta_path = run_lars(train_data, train_target, tol=-10000)
    sum_abs_coeff = np.sum(np.abs(beta_path), 1)
    # plotting code is based on:
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
    plt.plot(sum_abs_coeff, beta_path)
    plt.title('LARS Path')
    plt.ylabel('beta_j')
    plt.xlabel('sum_j(|coeff_j|)')
    plt.show()
