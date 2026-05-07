# eigen_solvers.py
import numpy as np

def power_method(A, v0=None, epsilon=1e-6, max_iter=1000):
    """
    幂法求主特征值（按模最大）
    返回: (特征值, 特征向量, 迭代次数, 是否收敛)
    """
    n = A.shape[0]
    if v0 is None:
        v0 = np.random.rand(n)
    v = v0 / np.max(np.abs(v0))
    for i in range(max_iter):
        u = A @ v
        m = u[np.argmax(np.abs(u))]   # 按模最大分量（保留符号）
        v_new = u / m
        if np.linalg.norm(v_new - v, np.inf) < epsilon:
            return m, v_new, i + 1, True
        v = v_new
    return m, v, max_iter, False

def inverse_power_method(A, v0=None, epsilon=1e-6, max_iter=1000):
    """
    逆幂法求按模最小特征值
    返回: (特征值, 特征向量, 迭代次数, 是否收敛)
    """
    n = A.shape[0]
    if v0 is None:
        v0 = np.random.rand(n)
    v = v0 / np.max(np.abs(v0))
    for i in range(max_iter):
        u = np.linalg.solve(A, v)
        m = u[np.argmax(np.abs(u))]
        v_new = u / m
        if np.linalg.norm(v_new - v, np.inf) < epsilon:
            return 1.0 / m, v_new, i + 1, True
        v = v_new
    return 1.0 / m, v, max_iter, False