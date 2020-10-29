"""
usage:
    python gpr.py data/gpr.dat --output img/gpr-simple.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, log, sqrt
from numpy.linalg import det, inv
from scipy.optimize import minimize, fmin_l_bfgs_b
from paramz.optimization.scg import SCG

# plot parameters
N     = 100
X_MIN = -1
X_MAX = 3.5
Y_MIN = -1
Y_MAX = 3
BLUE  = '#ccccff'


def kgauss(params):
    tau, sigma, eta = params
    return lambda x, y, train=True: \
        exp(tau) * exp(-(x - y)**2 / exp(sigma)) + \
        (exp(eta) if (train and x == y) else 0)


def kgauss_grad(xi, xj, d, kernel, params):
    if d == 0:
        return exp(params[d]) * kernel(params)(xi, xj)
    elif d == 1:
        return kernel(params)(xi, xj) * \
            (xi - xj) * (xi - xj) / exp(params[d])
    elif d == 2:
        return exp(params[d]) if xi == xj else 0
    else:
        return 0


def kv(x, x_train, kernel):
    return np.array([kernel(x, xi, False) for xi in x_train])


def kernel_matrix(xx, kernel):
    n = xx.size
    return np.array([
        kernel(xi, xj) for xi in xx for xj in xx
    ]).reshape(n, n)


def gpr(xx, x_train, y_train, kernel):
    K = kernel_matrix(x_train, kernel)
    K_inv = inv(K)

    y_pr = np.zeros(xx.shape)
    s_pr = np.zeros(xx.shape)

    for i, x in enumerate(xx):
        s = kernel(x, x)
        k = kv(x, x_train, kernel)
        y_pr[i] = k.T @ K_inv @ y_train
        s_pr[i] = s - k.T @ K_inv @ k

    return y_pr, s_pr


def tr(A, B):
    return (A * B.T).sum()


def print_param(params):
    print(params)


def loglik(params, x_train, y_train, kernel, kgrad):
    K = kernel_matrix(x_train, kernel(params))
    K_inv = inv(K)

    return log(det(K)) + y_train.T @ K_inv @ y_train
    # return (N * log(2 * np.pi) + log(det(K)) + y_train.T @ K_inv @ y_train)) / 2


def gradient(params, x_train, y_train, kernel, kgrad):
    K = kernel_matrix(x_train, kernel(params))
    K_inv = inv(K)
    K_inv_y = K_inv @ y_train

    D = len(params)
    n = x_train.shape[0]

    grad = np.zeros(D)

    for d in range(D):
        G = np.array([
            kgrad(xi, xj, d, kernel, params) for xi in x_train for xj in x_train
        ]).reshape(n, n)

        grad[d] = tr(K_inv, G) - K_inv_y @ G @ K_inv_y

    return grad


def numgrad(params, x_train, y_train, kernel, kgrad, eps=1e-6):
    D = len(params)
    ngrad = np.zeros(D)

    for d in range(D):
        lik = loglik(params, x_train, y_train, kernel, kgrad)
        params[d] += eps
        newlik = loglik(params, x_train, y_train, kernel, kgrad)
        params[d] -= eps
        ngrad[d] = (newlik - lik) / eps

    return ngrad


def optimize(x_train, y_train, kernel, kgrad, init):
    res = minimize(
        loglik,
        init,
        args=(x_train, y_train, kernel, kgrad),
        jac=gradient,  # numgrad
        method='BFGS',
        callback=print_param,
        options={'gtol': 1e-4, 'disp': True})
    print(res.message)

    return res.x


def optimize1(x_train, y_train, kernel, kgrad, init):
    x, flog, feval, status = SCG(
        loglik,
        gradient,
        init,
        optargs=[x_train, y_train, kernel, kgrad])
    print(f'status = {status}')

    return x


def optimize2(x_train, y_train, kernel, kgrad, init):
    x, f, d = fmin_l_bfgs_b(
        loglik,
        init,
        fprime=gradient,
        args=[x_train, y_train, kernel, kgrad],
        iprint=0,
        maxiter=1000)
    print(f'd = {d}')

    return x


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

    # kernel parameters
    tau   = log(1)
    sigma = log(1)
    eta   = log(1)

    kernel = kgauss
    kgrad  = kgauss_grad
    params = (tau, sigma, eta)
    params = optimize(x_train, y_train, kernel, kgrad, params)
    xx = np.linspace(X_MIN, X_MAX, N)

    print(f'grad   = {gradient(params, x_train, y_train, kernel, kgrad)}')
    print(f'ngrad  = {numgrad (params, x_train, y_train, kernel, kgrad)}')
    print(f'params = {params}')

    gpplot(xx, x_train, y_train, kernel, params, args.output)


if __name__ == "__main__":
    main()
