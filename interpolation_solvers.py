import numpy as np
import sympy as sp

def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    """拉格朗日插值算法"""
    n = len(x_nodes)
    m = len(x_eval)
    y_eval = np.zeros(m)
    
    for i in range(m):
        xi = x_eval[i]
        s = 0.0
        for j in range(n):
            # 计算基函数 L_j(x)
            p = 1.0
            for k in range(n):
                if k != j:
                    p *= (xi - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
            s += y_nodes[j] * p
        y_eval[i] = s
    return y_eval

def newton_interpolation(x_nodes, y_nodes, x_eval):
    """牛顿插值算法(均差法)"""
    n = len(x_nodes)
    # 计算均差表
    coef = np.zeros((n, n))
    coef[:, 0] = y_nodes
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x_nodes[i + j] - x_nodes[i])
            
    # 提取多项式系数（均差表的首行）
    a = coef[0, :]
    
    # 计算评估点的值
    m = len(x_eval)
    y_eval = np.zeros(m)
    for i in range(m):
        xi = x_eval[i]
        val = a[0]
        p = 1.0
        for j in range(1, n):
            p *= (xi - x_nodes[j - 1])
            val += a[j] * p
        y_eval[i] = val
    return y_eval



def best_square_approximation_cubic():
    """
    针对 f(x) = 1/(1+9x^2) 在 [-1, 1] 上的三次最佳平方逼近多项式
    使用 SymPy 进行精确积分计算法方程：H * c = b
    返回：多项式的 SymPy 表达式和 numpy 函数对象
    """
    x = sp.Symbol('x')
    f = 1 / (1 + 9 * x**2)
    
    # 基函数为 1, x, x^2, x^3
    phi = [1, x, x**2, x**3]
    n = len(phi)
    
    H = sp.Matrix.zeros(n, n)
    b = sp.Matrix.zeros(n, 1)
    
    # 构建希尔伯特型的法方程矩阵
    for i in range(n):
        for j in range(n):
            H[i, j] = sp.integrate(phi[i] * phi[j], (x, -1, 1))
        # 列下标必须是 0，不能写 1
        b[i, 0] = sp.integrate(f * phi[i], (x, -1, 1))
        
    # 求解线性方程组得到系数
    c = H.LUsolve(b)
    
    # 拼装多项式表达式
    poly_expr = c[0] * phi[0] + c[1] * phi[1] + c[2] * phi[2] + c[3] * phi[3]
    poly_expr = sp.simplify(poly_expr)
    
    # 转化为 numpy 可调用的函数
    poly_func = sp.lambdify(x, poly_expr, 'numpy')
    
    return poly_expr, poly_func


def universal_best_square_approximation(f_expr, x_var, a, b, degree=3):
    """
    针对任意函数 f_expr 在任意区间 [a, b] 上的任意degree次最佳平方逼近。
    参数：
        f_expr: SymPy 表达式 (例如 sp.sin(x_var) 或 1/(x_var**2 - 2))
        x_var: SymPy 符号变量 (例如 x)
        a, b: 区间下限和上限
        degree: 逼近多项式的最高次数 (默认为3)
    """
    if a >= b:
        raise ValueError("区间错误：必须满足 a < b")

    # 【安全锁】粗略探测奇点：检查端点和中点是否无定义(比如分母为0)
    try:
        f_expr.subs(x_var, a).evalf()
        f_expr.subs(x_var, (a + b) / 2).evalf()
        f_expr.subs(x_var, b).evalf()
    except Exception:
        raise ValueError(f"危险操作：函数在区间 [{a}, {b}] 内疑似存在奇点或无定义，拒绝计算！")

    # 1. 动态生成基函数：[1, x, x^2, ..., x^degree]
    phi = [x_var**k for k in range(degree + 1)]
    n = len(phi)
    
    H = sp.Matrix.zeros(n, n)
    B = sp.Matrix.zeros(n, 1)
    
    # 2. 积分域变成广义的 [a, b]
    for i in range(n):
        for j in range(n):
            # 构建克拉姆矩阵
            integral_val = sp.integrate(phi[i] * phi[j], (x_var, a, b))
            H[i, j] = integral_val
        
        # 构建右侧向量（加入一层防爆判定，防止符号积分算不出发散的瑕积分）
        try:
            b_val = sp.integrate(f_expr * phi[i], (x_var, a, b))
            # 检查结果是否包含 Infinity 或者 NaN
            if b_val.has(sp.oo, sp.zoo, sp.nan):
                raise ValueError
            B[i, 0] = b_val
        except ValueError:
            raise ValueError(f"积分发散：函数 {f_expr} 与 x^{i} 在 [{a}, {b}] 上的积分不存在！")
            
    # 3. 求解与拼装 (和以前一样)
    c = H.LUsolve(B)
    poly_expr = sp.simplify(sum(c[k] * phi[k] for k in range(n)))
    poly_func = sp.lambdify(x_var, poly_expr, 'numpy')
    
    return poly_expr, poly_func