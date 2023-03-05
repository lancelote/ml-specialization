from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

Vector: TypeAlias = npt.NDArray[np.float64]


def compute_model_output(x: Vector, w: int, b: int) -> Vector:
    (m,) = x.shape
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def main() -> None:
    x_train = np.array([1.0, 2.0])  # size 1k sqft
    y_train = np.array([300, 500])  # price 1k dollars
    print(f"{x_train=}")
    print(f"{y_train=}")

    (m,) = x_train.shape
    print(f"{m=}")

    i = 0
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

    plt.scatter(x_train, y_train, marker="x", c="r")
    plt.title("Housing Prices")
    plt.ylabel("Price (in 1000s of dollars)")
    plt.xlabel("Size (1000 sqft)")
    plt.show()

    w = 200
    b = 100
    print(f"{w=}, {b=}")

    tmp_f_wb = compute_model_output(x_train, w, b)

    plt.plot(x_train, tmp_f_wb, c="b", label="Our prediction")
    plt.scatter(x_train, y_train, marker="x", c="r", label="Actual values")
    plt.title("Housing Prices")
    plt.ylabel("Price (in 1000s of dollars")
    plt.xlabel("Size (1000 sqft)")
    plt.legend()
    plt.show()

    cost_1200sqft = w * 1.2 + b
    print(f"${cost_1200sqft:.0f} thousand dollars for 1200 sqft house")


if __name__ == "__main__":
    main()
