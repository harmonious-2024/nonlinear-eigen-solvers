# nonlinear_solvers.py
import numpy as np

def simple_iteration(phi, x0, epsilon=1e-6, max_iter=100):
    """
    简单迭代法：x_{k+1} = phi(x_k)
    返回: (近似根, 迭代次数, 是否收敛)
    """
    x = x0
    for i in range(max_iter):
        x_new = phi(x)
        if abs(x_new - x) < epsilon:
            return x_new, i + 1, True
        x = x_new
    return x, max_iter, False


def steffensen(phi, x0, epsilon=1e-6, max_iter=100):
    """
    Steffensen 加速法 (Aitken Δ² 过程)
    返回: (近似根, 迭代次数, 是否收敛)
    """
    x = x0
    for i in range(max_iter):
        y = phi(x)
        z = phi(y)
        denominator = z - 2 * y + x
        if abs(denominator) < 1e-15:
            return x, i + 1, False
        x_new = x - (y - x) ** 2 / denominator
        if abs(x_new - x) < epsilon:
            return x_new, i + 1, True
        x = x_new
    return x, max_iter, False


def newton(f, df, x0, epsilon=1e-6, max_iter=100):
    """
    牛顿迭代法：x_{k+1} = x_k - f(x_k)/f'(x_k)
    返回: (近似根, 迭代次数, 是否收敛)
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            return x, i + 1, False
        x_new = x - fx / dfx
        if abs(x_new - x) < epsilon:
            return x_new, i + 1, True
        x = x_new
    return x, max_iter, False


def newton_downhill(f, df, x0, epsilon=1e-6, max_iter=100,
                    epsilon_lambda=1e-8, max_downhill=20, alpha=0.5):
    """
    牛顿下山法
    返回: (近似根, 迭代次数, 各步下山因子列表, 是否成功)
    """
    x = x0
    f0 = f(x)
    lambdas = []
    for i in range(max_iter):
        dfx = df(x)
        if abs(dfx) < 1e-15:
            return x, i + 1, lambdas, False
        lam = 1.0
        # 下山搜索
        for _ in range(max_downhill):
            x1 = x - lam * f0 / dfx
            f1 = f(x1)
            if abs(f1) < abs(f0):  # 下降条件
                break
            if lam < epsilon_lambda:  # 下山失败
                return x, i + 1, lambdas, False
            lam *= alpha
        else:
            return x, i + 1, lambdas, False

        lambdas.append(lam)
        if abs(f1) < epsilon or abs(x1 - x) < epsilon:
            return x1, i + 1, lambdas, True
        x, f0 = x1, f1
    return x, max_iter, lambdas, False