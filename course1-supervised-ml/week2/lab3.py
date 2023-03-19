import copy

import matplotlib.pyplot as plt
import numpy as np


def compute_cost(x, y, w, b):
    return np.sum((np.dot(x, w) + b - y) ** 2)


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    error = np.dot(x, w) + b - y
    dj_dw = np.sum(np.transpose(x) * error, axis=1) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    hist = {"cost": []}
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        hist["cost"].append(compute_cost(x, y, w, b))

    return w, b, hist


def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=",", skiprows=1)
    x = data[:, :4]
    y = data[:, 4]
    return x, y


def plot_features_by_price(x_train, y_train, x_labels):
    fig, ax = plt.subplots(1, 4, sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:, i], y_train)
        ax[i].set_xlabel(x_labels[i])
    ax[0].set_ylabel("price (1000's)")
    plt.show()


def plot_cost(hist):
    fig, ax = plt.subplots()
    ax.plot(range(len(hist["cost"])), hist["cost"])
    ax.set_title("cost vs. iterations")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    ax.grid()
    plt.show()


def print_results(w, b, hist):
    step = len(hist["cost"]) // 10
    for i in range(10):
        cost = hist["cost"][i * step]
        print(f"{i * step}: {cost}")
    print(f"\n{w=}, {b=}\n")


def zscore_normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


def plot_normalized(x_train, x_norm, x_labels):
    fix, ax = plt.subplots(1, 2)

    ax[0].scatter(x_train[:, 0], x_train[:, 3])
    ax[0].set_xlabel(x_labels[0])
    ax[0].set_ylabel(x_labels[3])
    ax[0].set_title("unnormalized")
    ax[0].axis("equal")

    ax[1].scatter(x_norm[:, 0], x_norm[:, 3])
    ax[1].set_xlabel(x_labels[0])
    ax[1].set_ylabel(x_labels[3])
    ax[1].set_title("Z-score normalized")
    ax[1].axis("equal")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_predict(x_train, y_train, x_norm, w_norm, b_norm, x_labels):
    m = x_norm.shape[0]
    y_pred = np.zeros(m)

    for i in range(m):
        y_pred[i] = np.dot(x_norm[i], w_norm) + b_norm

    fix, ax = plt.subplots(1, 4, sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train[:, i], y_train, label="target")
        ax[i].set_xlabel(x_labels[i])
        ax[i].scatter(x_train[:, i], y_pred, label="predict", color="#FF9300")
    ax[0].set_ylabel("price")
    plt.show()


def predict_house(x_train, w_norm, b_norm, size, bed, floor, age):
    x_house = np.array([1200, 3, 1, 40])
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    x_house_norm = (x_house - mu) / sigma
    print(x_house_norm)
    x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
    print(
        f"{size} sqft house,"
        f" {bed} bedrooms,"
        f" {floor} floor,"
        f" {age} years old"
        f" = ${x_house_predict * 1000:0.0f}"
    )


def main():
    x_train, y_train = load_house_data()
    _, n = x_train.shape
    x_labels = ["size (sqft)", "bedrooms", "floors", "age"]

    plot_features_by_price(x_train, y_train, x_labels)

    # run gradient descent
    w, b, hist = gradient_descent(
        x_train, y_train, w_in=np.zeros(n), b_in=0, alpha=1e-7, num_iters=10
    )
    plot_cost(hist)
    print_results(w, b, hist)

    # scaling
    x_norm = zscore_normalize(x_train)
    plot_normalized(x_train, x_norm, x_labels)

    # rerun GD on scaled features
    w_norm, b_norm, hist = gradient_descent(
        x_norm, y_train, w_in=np.zeros(n), b_in=0, alpha=1e-1, num_iters=1000
    )
    plot_cost(hist)
    print_results(w_norm, b_norm, hist)

    # make predictions
    plot_predict(x_train, y_train, x_norm, w_norm, b_norm, x_labels)
    predict_house(x_train, w_norm, b_norm, size=1200, bed=3, floor=1, age=40)


if __name__ == "__main__":
    main()
