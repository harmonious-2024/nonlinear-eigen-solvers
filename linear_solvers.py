import numpy as np
from typing import Tuple, Optional


def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对矩阵A进行Doolittle LU分解：A = L * U
    """
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("LU分解矩阵必须是方阵")

    L = np.eye(n, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)

    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        if abs(U[k, k]) < 1e-15:
            raise ValueError(f"主元 U[{k},{k}] 接近于0，LU分解失败。")
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]

    return L, U


def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    使用LU分解求解线性方程组 Ax = b
    """
    L, U = lu_decomposition(A)
    n = A.shape[0]

    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]

    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return x


def jacobi_iteration(A: np.ndarray, b: np.ndarray,
                     x0: Optional[np.ndarray] = None,
                     tol: float = 1e-6,
                     max_iter: int = 1000) -> Tuple[np.ndarray, int, bool]:
    """
    Jacobi迭代法求解 Ax = b
    """
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)

    x = x0.copy()
    x_new = np.zeros_like(x)
    converged = False

    if np.any(np.abs(np.diag(A)) < 1e-15):
        raise ValueError("矩阵对角线包含接近0的元素，无法使用Jacobi迭代。")

    for k in range(max_iter):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            converged = True
            break
        x = x_new.copy()

    return x_new, k + 1, converged


def gauss_seidel_iteration(A: np.ndarray, b: np.ndarray,
                           x0: Optional[np.ndarray] = None,
                           tol: float = 1e-6,
                           max_iter: int = 1000) -> Tuple[np.ndarray, int, bool]:
    """
    Gauss-Seidel迭代法求解 Ax = b
    """
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)

    x = x0.copy()
    converged = False

    if np.any(np.abs(np.diag(A)) < 1e-15):
        raise ValueError("矩阵对角线包含接近0的元素，无法使用Gauss-Seidel迭代。")

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x_old[i + 1:]
            x[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            converged = True
            break

    return x, k + 1, converged


def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-6,
                       max_iter: int = 1000) -> Tuple[np.ndarray, int, bool]:
    """
    共轭梯度法(CG)求解 Ax = b
    """
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)

    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    converged = False

    for k in range(max_iter):
        Ap = A @ p
        pAp = p.T @ Ap
        if abs(pAp) < 1e-15:
            break

        alpha = (r.T @ r) / pAp
        x = x + alpha * p
        r_new = r - alpha * Ap

        if np.linalg.norm(r_new) < tol:
            converged = True
            break

        beta = (r_new.T @ r_new) / (r.T @ r)
        p = r_new + beta * p
        r = r_new

    return x, k + 1, converged