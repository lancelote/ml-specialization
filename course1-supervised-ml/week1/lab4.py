import math
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Vector: TypeAlias = npt.NDArray[np.float64]
Gradient: TypeAlias = tuple[float, float]
Coefficients: TypeAlias = tuple[float, float]
CostFunction: TypeAlias = Callable[[Vector, Vector, float, float], float]
GradientFunction: TypeAlias = Callable[[Vector, Vector, float, float], Gradient]


def compute_cost(x: Vector, y: Vector, w: float, b: float) -> float:
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


def compute_gradient(x: Vector, y: Vector, w: float, b: float) -> Gradient:
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(
    x: Vector,
    y: Vector,
    w_in: float,
    b_in: float,
    alpha: float,
    n_iters: int,
    cost_function: CostFunction,
    gradient_function: GradientFunction,
) -> Coefficients:
    w = w_in
    b = b_in

    for i in range(n_iters):
        cost = cost_function(x, y, w, b)
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % math.ceil(n_iters / 10) == 0:
            print(
                f"Iteration: {i:4}: Cost {cost:0.2e} ",
                f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e} ",
                f"w: {w:0.3e}, b:{b:0.5e}",
            )

    return w, b


def main() -> None:
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    w_init = 0
    b_init = 0

    iterations = 10_000
    tmp_alpha = 1.0e-2

    w_final, b_final = gradient_descent(
        x_train,
        y_train,
        w_init,
        b_init,
        tmp_alpha,
        iterations,
        compute_cost,
        compute_gradient,
    )
    print(f"(w,b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")


if __name__ == "__main__":
    main()
