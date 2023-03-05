from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Vector: TypeAlias = npt.NDArray[np.float64]


def compute_cost(x: Vector, y: Vector, w: int, b: int) -> float:
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


def main() -> None:
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    w = 200
    b = 100

    cost = compute_cost(x_train, y_train, w, b)

    print(f"Cost at {w=} and {b=} is {cost}")


if __name__ == "__main__":
    main()
