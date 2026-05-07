# equations.py
import numpy as np

def f(x):
    """方程 f(x) = x^3 + 2x^2 + 10x - 20"""
    return x**3 + 2*x**2 + 10*x - 20

def df(x):
    """f(x) 的导数 f'(x) = 3x^2 + 4x + 10"""
    return 3*x**2 + 4*x + 10

def phi(x):
    """简单迭代法 / Steffensen 法使用的迭代函数
    φ(x) = 20 / (x^2 + 2x + 10)
    （由原方程等价变形得到，在 [0,2] 内收敛）
    """
    return 20.0 / (x**2 + 2*x + 10)

def generate_matrix(n=4, seed=None):
    """生成测试用随机方阵，保证存在实特征值"""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    # 使矩阵不完全随机，增加主特征值显著性（可选）
    A = A + np.diag(np.sum(np.abs(A), axis=1))
    return A