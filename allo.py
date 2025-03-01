import numpy as np
import sympy as sp

# 変数の定義
x1, x2, y1, y2, P, r = sp.symbols("x1 x2 y1 y2 P r")
lambda1, lambda2 = sp.symbols("lambda1 lambda2")

# 関数の定義
N = -y1 * x1**2 + (1 + y1) * x1 - y2 * x2**2 + (1 + y2) * x2
D = (1 - y1) * x1 + y1 + (1 - y2) * x2 + y2
f = N / D  # 目的関数

# ラグランジュ関数の定義
L_hat = f - lambda1 * (P - x1 - x2) - lambda2 * (r - y1 - y2)

# fの偏微分
df_dx1 = sp.diff(f, x1)
df_dx2 = sp.diff(f, x2)
df_dy1 = sp.diff(f, y1)
df_dy2 = sp.diff(f, y2)

# Python関数に変換
grad_f_p = sp.lambdify((x1, x2, y1, y2), [df_dx1, df_dx2])
grad_f_q = sp.lambdify((x1, x2, y1, y2), [df_dy1, df_dy2])


# lambda_star 関数の再定義
def lambda_star(p, q):
    # L_hatの偏微分を計算
    dL_dx1_value = sp.diff(L_hat, x1).subs(
        {x1: p[0], x2: p[1], y1: q[0], y2: q[1], lambda1: 0, lambda2: 0, P: 2, r: 2}
    )
    dL_dy1_value = sp.diff(L_hat, y1).subs(
        {x1: p[0], x2: p[1], y1: q[0], y2: q[1], lambda1: 0, lambda2: 0, P: 2, r: 2}
    )

    lambda1_value = -dL_dx1_value.evalf()
    lambda2_value = -dL_dy1_value.evalf()

    return np.array([float(lambda1_value), float(lambda2_value)])


def gradient_g(p, q):
    # gの勾配（例：定数）
    return np.array([-1.0, 0.0])  # 実際の問題に応じて変更が必要


def projection_onto_set(x, set_bounds):
    # 値を指定された範囲にクリップする
    return np.clip(x, set_bounds[0], set_bounds[1])


def ms_gd_optimization(P_bounds, Q_bounds, p0, q0, eta, eta_prime, Tp, Tq, decay=True):
    # 初期化
    p = np.array(p0, dtype=float)
    q = np.array(q0, dtype=float)
    p_hist = []
    q_hist = []

    for t in range(1, Tp + 1):
        # 50回ごとに進行状況を表示
        if not t % 50:
            print(f" ----- Iteration {t}/{Tp} ----- ")

        # 履歴に現在のpとqの値を保存
        p_hist.append(np.copy(p))
        q_hist.append(np.copy(q))

        # 内部ループ：qの更新
        for s in range(1, Tq + 1):
            # qに関する勾配を計算
            grad_q = np.array(grad_f_q(p[0], p[1], q[0], q[1]), dtype=float)
            # qの更新と射影
            q = projection_onto_set(q + eta_prime * grad_q.flatten(), Q_bounds)

        # λ*の計算
        lambdas = lambda_star(p, q)
        # pに関する勾配を計算
        grad_p = np.array(grad_f_p(p[0], p[1], q[0], q[1]), dtype=float)
        # 勾配の総和を計算
        gradient_sum = grad_p.flatten() + np.sum(
            [l * gradient_g(p, q) for l in lambdas], axis=0
        )

        # ステップサイズの計算
        if decay:
            step_size = (t ** (-1 / 2)) * gradient_sum
        else:
            step_size = eta * gradient_sum

        # pの更新
        p += step_size * (p > 0)
        # pの射影
        p = projection_onto_set(p, P_bounds)

    return p, q, p_hist, q_hist


# Example parameters
P_bounds = (0, 1)  # pの範囲
Q_bounds = (0, 1)  # qの範囲
eta = 0.01
eta_prime = 0.01
Tp = 1000
Tq = 100
p0 = [0.0, 0.0]  # pの初期値
q0 = [0.0, 0.0]  # qの初期値

# 最適化アルゴリズムの実行
p_final, q_final, p_hist, q_hist = ms_gd_optimization(
    P_bounds, Q_bounds, p0, q0, eta, eta_prime, Tp, Tq
)

print(f"Final p: {p_final}")
print(f"Final q: {q_final}")
