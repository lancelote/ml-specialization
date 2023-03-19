import copy

import numpy as np


def predict(x, w, b: int):
    return np.dot(x, w) + b


def compute_cost(x, y, w, b):
    m = len(x)
    return np.sum((np.dot(x, w) + b - y) ** 2) / (2 * m)

    # m = x.shape[0]
    # cost = .0
    #
    # for i in range(m):
    #     f_wb_i = np.dot(x[i], w) + b
    #     cost += (f_wb_i - y[i])**2
    #
    # cost /= (2 * m)
    # return cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    error = np.dot(x, w) + b - y
    dj_dw = np.sum(np.transpose(x) * error, axis=1) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

    # m, n = x.shape
    # dj_dw = np.zeros((n,))
    # dj_db = .0
    #
    # for i in range(m):
    #     error = np.dot(x[i], w) + b - y[i]
    #     for j in range(n):
    #         dj_dw[j] += error * x[i, j]
    #     dj_db += error
    #
    # dj_dw /= m
    # dj_db /= m
    # return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b


def main():
    # size, bedrooms, floors, age
    x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])

    # price
    y = np.array([460, 232, 178])

    w_init = np.zeros(x.shape[1])
    b_init = 0.0
    iters = 1_000
    alpha = 5.0e-7

    w_final, b_final = gradient_descent(x, y, w_init, b_init, alpha, iters)
    print(f"found: w={w_final}, b={b_final:0.2f}")


if __name__ == "__main__":
    main()
