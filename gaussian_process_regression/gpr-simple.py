"""
usage:
    python gpr-simple.py data/gpr.dat --output img/gpr.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt
from numpy.linalg import inv

# plot parameters
N     = 100
X_MIN = -1
X_MAX = 3.5
Y_MIN = -1
Y_MAX = 3
BLUE  = '#ccccff'

# GP kernel parameters
ETA   = 0.1
TAU   = 1
SIGMA = 1


def kgauss(params):
    tau, sigma = params
    return lambda x, y: tau * exp(-(x - y) ** 2 / (2 * sigma * sigma))


def kv(x, x_train, kernel):
    return np.array([kernel(x, xi) for xi in x_train])


def kernel_matrix(x, kernel):
    n = x.size
    return np.array([
        kernel(xi, xj) for xi in x for xj in x
    ]).reshape(n, n) + ETA * np.eye(n)


def gpr(xx, x_train, y_train, kernel):
    K = kernel_matrix(x_train, kernel)
    K_inv = inv(K)

    y_pr = np.zeros(xx.shape)
    s_pr = np.zeros(xx.shape)

    for i, x in enumerate(xx):
        s = kernel(x, x) + ETA
        k = kv(x, x_train, kernel)
        y_pr[i] = k.T @ K_inv @ y_train
        s_pr[i] = s - k.T @ K_inv @ k

    return y_pr, s_pr


def gpplot(xx, x_train, y_train, kernel, params, filename=None):
    y_pr, s_pr = gpr(xx, x_train, y_train, kernel(params))

    plt.ylim(Y_MIN, Y_MAX)
    plt.xlim(X_MIN, X_MAX)
    plt.plot(x_train, y_train, 'bx', markersize=16)
    plt.plot(xx, y_pr, 'b-')
    plt.fill_between(
        xx,
        y_pr - 2 * sqrt(s_pr),
        y_pr + 2 * sqrt(s_pr),
        color=BLUE
    )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="load this file as data source")
    parser.add_argument("-o", "--output", help="image name to be saved")
    args = parser.parse_args()

    train = np.loadtxt(args.data_path, dtype=float)
    x_train = train[:, 0]
    y_train = train[:, 1]

    kernel = kgauss
    params = (TAU, SIGMA)
    xx = np.linspace(X_MIN, X_MAX, N)

    gpplot(xx, x_train, y_train, kernel, params, args.output)


if __name__ == "__main__":
    main()
