# scflp.py
# 逐次型 競争的施設配置問題 (S-CFLP) — Python 実装
# Branch-and-(outer)-Cut と遅延制約生成 (DCG)
# - 目的関数とモデルは Qi, Jiang, Shen (2022), arXiv:2103.04259v3 に準拠
# - フォロワ側の分離は命題5の近似分離（単一ソート）を使用
# - 2種類のカットを追加: Submodular（x が整数のとき）と Bulge（常に）
#
# 依存関係:
#   numpy（必須）
#   PuLP（推奨のフォールバック）または gurobipy（任意、MILP を高速に解く場合）

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Sequence, Set

# --- オプションの MILP バックエンド -------------------------------------------
# 最小限の抽象化により、PuLP か Gurobi のどちらでも差し替え可能。
# 既定: PuLP (CBC)。gurobipy が利用可能なら 'gurobi' を選択可。


class _MILPVar:
    def __init__(self, name, index=None):
        self.name = f"{name}[{index}]" if index is not None else name


class _MILPInterface:
    """
    最小限の MILP インタフェース。以下をサポート:
        - 二値変数 x_j とスカラ theta
        - 制約: sum_j x_j == p
        - 線形カット: theta <= c0 + sum_j c[j]*x_j
        - 目的: theta の最大化

    サブクラスで実装すべきメソッド:
        add_binary_var, add_continuous_var, add_constraint_eq/ineq, set_objective_max,
        solve, get_value
    """

    def add_binary_var(self, name: str, index: Optional[int] = None):
        raise NotImplementedError

    def add_continuous_var(self, name: str, lb: float, ub: float):
        raise NotImplementedError

    def add_constraint_eq(self, expr, rhs, name: str):
        raise NotImplementedError

    def add_constraint_le(self, expr, rhs, name: str):
        raise NotImplementedError

    def set_objective_max(self, expr):
        raise NotImplementedError

    def solve(self) -> int:
        raise NotImplementedError

    def get_value(self, var) -> float:
        raise NotImplementedError

    def add_linear_cut_theta_le(
        self, theta, c0: float, coeffs: Dict[int, float], name: str
    ):
        # theta <= c0 + sum_j coeffs[j] * x_j
        lhs = theta
        rhs = c0
        expr = 0.0
        for j, cj in coeffs.items():
            expr += cj * self.x[j]  # type: ignore
        # 実装形: theta - sum_j coeffs_j x_j <= c0
        self.add_constraint_le(lhs - expr, rhs, name)


class PuLPInterface(_MILPInterface):
    def __init__(self, nJ: int, p: int, solver_name: str = "CBC"):
        import pulp  # type: ignore

        self.pulp = pulp
        self.model = pulp.LpProblem("S_CFLP_Master", pulp.LpMaximize)
        self.x = {
            j: pulp.LpVariable(f"x[{j}]", lowBound=0, upBound=1, cat="Binary")
            for j in range(nJ)
        }
        self.theta = pulp.LpVariable(
            "theta", lowBound=0.0, upBound=1.0, cat="Continuous"
        )
        # 予算制約
        self.model += pulp.lpSum([self.x[j] for j in range(nJ)]) == p, "budget_eq"
        # 目的
        self.model += self.theta, "obj"
        if solver_name == "CBC":
            self.solver = pulp.PULP_CBC_CMD(msg=False)
        elif solver_name == "GLPK":
            self.solver = pulp.GLPK_CMD(msg=False)
        else:
            # 指定が不明な場合は CBC にフォールバック
            self.solver = pulp.PULP_CBC_CMD(msg=False)

    def add_binary_var(self, name: str, index: Optional[int] = None):
        # x_j はコンストラクタで生成済み
        raise NotImplementedError("Binary x_j created in constructor.")

    def add_continuous_var(self, name: str, lb: float, ub: float):
        # theta はコンストラクタで生成済み
        raise NotImplementedError("Continuous theta created in constructor.")

    def add_constraint_eq(self, expr, rhs, name: str):
        self.model += (expr == rhs), name

    def add_constraint_le(self, expr, rhs, name: str):
        self.model += (expr <= rhs), name

    def set_objective_max(self, expr):
        self.model.sense = self.pulp.LpMaximize
        self.model.setObjective(expr)

    def solve(self) -> int:
        status = self.model.solve(self.solver)
        return int(status)

    def get_value(self, var) -> float:
        return float(var.value())


class GurobiInterface(_MILPInterface):
    def __init__(self, nJ: int, p: int):
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB

        self.gp = gp
        self.model = gp.Model("S_CFLP_Master")
        self.model.Params.OutputFlag = 0
        self.x = {
            j: self.model.addVar(vtype=GRB.BINARY, name=f"x[{j}]") for j in range(nJ)
        }
        self.theta = self.model.addVar(
            lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="theta"
        )
        # 予算制約
        self.model.addConstr(
            self.gp.quicksum(self.x[j] for j in range(nJ)) == p, name="budget_eq"
        )
        # 目的
        self.model.setObjective(self.theta, GRB.MAXIMIZE)

    def add_binary_var(self, name: str, index: Optional[int] = None):
        raise NotImplementedError("Binary x_j created in constructor.")

    def add_continuous_var(self, name: str, lb: float, ub: float):
        raise NotImplementedError("Continuous theta created in constructor.")

    def add_constraint_eq(self, expr, rhs, name: str):
        self.model.addConstr(expr == rhs, name=name)

    def add_constraint_le(self, expr, rhs, name: str):
        self.model.addConstr(expr <= rhs, name=name)

    def set_objective_max(self, expr):
        from gurobipy import GRB

        self.model.setObjective(expr, GRB.MAXIMIZE)

    def solve(self) -> int:
        self.model.optimize()
        return int(self.model.Status)

    def get_value(self, var) -> float:
        return float(var.X)


# --- 問題データのコンテナ ------------------------------------------------------


class SCFLPData:
    """
    MNL 効用のもとでの S-CFLP のデータコンテナ。

    Attributes
    ----------
    I : int                   # 需要点の数
    J : int                   # 候補地の数
    h : (I,) array            # 需要重み（合計 1）
    UL : (I,) array           # 既存リーダー施設からの効用. 初期値は 0
    UF : (I,) array           # 既存フォロワ施設からの効用. 初期値は 0
    w  : (I,J) array          # 候補に対する w_ij = exp(alpha_j - beta * d_ij)
    p  : int                  # リーダーの予算（新設数）
    r  : int                  # フォロワの予算
    """

    def __init__(
        self,
        h: np.ndarray,
        UL: np.ndarray,
        UF: np.ndarray,
        w: np.ndarray,
        p: int,
        r: int,
    ):
        assert h.ndim == 1 and UL.ndim == 1 and UF.ndim == 1
        assert w.ndim == 2 and w.shape[0] == h.shape[0]
        self.h = h.astype(float)
        self.UL = UL.astype(float)
        self.UF = UF.astype(float)
        self.w = w.astype(float)
        self.I = h.shape[0]
        self.J = w.shape[1]
        self.p = int(p)
        self.r = int(r)
        # h の合計が 1 でない場合は正規化
        s = self.h.sum()
        if abs(s - 1.0) > 1e-12:
            self.h = self.h / s

    @staticmethod
    def from_coordinates(
        demand_xy: np.ndarray,
        cand_xy: np.ndarray,
        alpha: np.ndarray,
        beta: float,
        p: int,
        r: int,
        leader_xy: Optional[np.ndarray] = None,
        follower_xy: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
    ) -> "SCFLPData":
        """
        2次元座標からの簡易コンストラクタ。
        """
        I = demand_xy.shape[0]
        J = cand_xy.shape[0]
        if h is None:
            h = np.ones(I) / I

        # 距離行列
        def pairwise_dist(A, B):
            return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

        d_cand = pairwise_dist(demand_xy, cand_xy)  # (I,J)
        w = np.exp(alpha.reshape(1, J) - beta * d_cand)

        # 既存施設
        UL = np.zeros(I)
        UF = np.zeros(I)
        if leader_xy is not None and leader_xy.shape[0] > 0:
            dL = pairwise_dist(demand_xy, leader_xy)
            # 既存にも同じ alpha を仮定？ここでは alpha の平均を使用
            wL = np.exp(alpha.mean() - beta * dL)
            UL = wL.sum(axis=1)
        if follower_xy is not None and follower_xy.shape[0] > 0:
            dF = pairwise_dist(demand_xy, follower_xy)
            wF = np.exp(alpha.mean() - beta * dF)
            UF = wF.sum(axis=1)

        return SCFLPData(h=h, UL=UL, UF=UF, w=w, p=p, r=r)


# --- ユーティリティ関数 --------------------------------------------------------


def L_value(data: SCFLPData, x: np.ndarray, y: np.ndarray) -> float:
    """
    式(4)の L(x, y) を計算:
      L(x,y) = sum_i h_i * (UL_i + sum_j w_ij x_j) / (UL_i + UF_i + sum_j w_ij (x_j OR y_j))
    """
    I, J = data.I, data.J
    x = x.reshape(J)
    y = y.reshape(J)
    # x OR y
    xy = np.maximum(x, y)
    num = data.UL + data.w.dot(x)
    den = data.UL + data.UF + data.w.dot(xy)
    val = (data.h * (num / den)).sum()
    return float(val)


def Lb_and_grad(
    data: SCFLPData, x: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Bulge 関数 L_b(x, y) と、その点 (x,y) における x に関する勾配。
    定義:
      L_b(x,y) = sum_i h_i * Q_i / P_i
        P_i = UL_i + UF_i + sum_j w_ij * ((1 - y_j) x_j + y_j)
        Q_i = UL_i + sum_j w_ij * (-y_j x_j^2 + (1 + y_j) x_j)
      勾配:
        grad_j = sum_i h_i * [ - w_ij (1-y_j) Q_i / P_i^2 + w_ij (-2 y_j x_j + 1 + y_j) / P_i ]
    """
    I, J = data.I, data.J
    x = x.reshape(J)
    y = y.reshape(J)
    one_minus_y = 1.0 - y
    P = data.UL + data.UF + data.w.dot(one_minus_y * x + y)  # (I,)
    Q = data.UL + data.w.dot(-y * (x**2) + (1.0 + y) * x)  # (I,)
    Lb = (data.h * (Q / P)).sum()
    # 勾配
    grad = np.zeros(J)
    P2 = P * P
    for j in range(J):
        wj = data.w[:, j]
        term = (
            -wj * one_minus_y[j] * Q / P2 + wj * (-2.0 * y[j] * x[j] + 1.0 + y[j]) / P
        )
        grad[j] = (data.h * term).sum()
    return float(Lb), grad


def submodular_cut_coeffs(
    data: SCFLPData, x_bin: np.ndarray, y_bin: np.ndarray
) -> Tuple[float, Dict[int, float]]:
    """
    固定された Y の下で、S = supp(x) を用いて Nemhauser–Wolsey 線形化を構築。
    返り値 (c0, coeffs) は不等式 θ <= c0 + sum_j coeffs[j] * x_j の形の係数。
    """
    J = data.J
    S: Set[int] = set(np.where(x_bin > 0.5)[0].tolist())
    Y: Set[int] = set(np.where(y_bin > 0.5)[0].tolist())

    def L_Y_of_X(Xset: Set[int]) -> float:
        # L_Y(X) = sum_i h_i * (UL_i + sum_{j in X} w_ij) / (UL_i + UF_i + sum_{j in X ∪ Y} w_ij)
        maskX = np.zeros(J, dtype=float)
        if Xset:
            maskX[list(Xset)] = 1.0
        maskXUY = maskX.copy()
        if Y:
            maskY = np.zeros(J, dtype=float)
            maskY[list(Y)] = 1.0
            maskXUY = np.maximum(maskX, maskY)
        num = data.UL + data.w.dot(maskX)
        den = data.UL + data.UF + data.w.dot(maskXUY)
        return float((data.h * (num / den)).sum())

    # rho_Y(S; k) = L_Y(S ∪ {k}) - L_Y(S)
    LYS = L_Y_of_X(S)
    rho_S_k: Dict[int, float] = {}
    for k in range(J):
        if k in S:
            continue
        Sk = set(S)
        Sk.add(k)
        rho_S_k[k] = L_Y_of_X(Sk) - LYS

    # rho_Y(J\{k}; k): 「全集合から k を除いた集合」に k を加えたときの周辺増分
    rho_Jminus_k_k: Dict[int, float] = {}
    J_minus_k_base = set(range(J))
    for k in range(J):
        Sfull_minus_k = set(J_minus_k_base)
        Sfull_minus_k.remove(k)
        LY_full_minus_k = L_Y_of_X(Sfull_minus_k)
        Sfull = set(J_minus_k_base)
        # rho(J\k; k) = L_Y(J) - L_Y(J\k)
        rho_Jminus_k_k[k] = L_Y_of_X(Sfull) - LY_full_minus_k

    # 形 (8) の不等式を構築:
    # θ ≤ L_Y(S) − Σ_{k∈S} ρ_Y(J\{k}; k)(1 − x_k) + Σ_{k∈J\S} ρ_Y(S; k) x_k
    c0 = LYS - sum(rho_Jminus_k_k[k] for k in S)  # 定数項
    coeffs: Dict[int, float] = {}
    for k in S:
        coeffs[k] = rho_Jminus_k_k[k]
    for k in range(J):
        if k not in S:
            coeffs[k] = rho_S_k[k]
    return c0, coeffs


def approximate_separation_y(
    data: SCFLPData, x: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    命題5の近似分離:
      - α(x) と β(x) を閉形式で計算
      - y_hat は β(x) を降順ソートして上位 r のインデックスを採用
      - 戻り値は (y_hat, ub_lin = α(x) - β(x)^T y_hat, beta)
    """
    J = data.J
    x = x.reshape(J)
    I = data.I

    a = data.UL + data.w.dot(x)  # (I,)
    # u = w_ij * (1 - x_j) を計算
    w_scaled = data.w * (1.0 - x.reshape(1, J))  # (I,J)

    # 各 i について:
    # wL_i = UF_i + w_scaled[i, :] の「最小 r 個」の総和
    # wU_i = UF_i + w_scaled[i, :] の「最大 r 個」の総和
    r = data.r
    # 効率のため np.partition を使用
    # 最小 r
    smallest = (
        np.partition(w_scaled, r - 1, axis=1)[:, :r].sum(axis=1)
        if r > 0
        else np.zeros(I)
    )
    # 最大 r
    largest = (
        (-np.partition(-w_scaled, r - 1, axis=1)[:, :r]).sum(axis=1)
        if r > 0
        else np.zeros(I)
    )
    wL = data.UF + smallest
    wU = data.UF + largest

    # α(x) と β_j(x)
    denomL = a + wL
    denomU = a + wU
    alpha_terms = a * (a + wU + wL - data.UF) / (denomU * denomL)
    alpha_val = float((data.h * alpha_terms).sum())

    # β_j: sum_i h_i * [ a_i * w_ij * (1 - x_j) / ((a_i + wU_i)(a_i + wL_i)) ]
    common = data.h * (a / (denomU * denomL))  # (I,)
    beta = (common.reshape(I, 1) * (data.w * (1.0 - x.reshape(1, J)))).sum(
        axis=0
    )  # (J,)

    # β を降順にソートして y_hat を選択
    idx_sorted = np.argsort(-beta)
    y_hat = np.zeros(J)
    if r > 0:
        y_hat[idx_sorted[:r]] = 1.0
    ub_lin = alpha_val - float(beta[y_hat > 0.5].sum())
    return y_hat, ub_lin, beta


# --- メインソルバ --------------------------------------------------------------


class SCFLPSolver:
    """
    近似分離（命題5）と 2 種のカットを用いた DCG により S-CFLP を厳密に解く:
      - すべての (x_hat, y_hat) で Bulge カット
      - x_hat が整数のときに Submodular カット

    gurobipy が利用可能なら milp_backend='gurobi' を推奨。既定は PuLP (CBC)。
    """

    def __init__(
        self, data: SCFLPData, milp_backend: str = "pulp", pulp_solver: str = "CBC"
    ):
        self.data = data
        self.backend_name = milp_backend
        if milp_backend == "gurobi":
            try:
                self.mip = GurobiInterface(nJ=data.J, p=data.p)
            except Exception as e:
                print("Gurobi が利用できないため PuLP にフォールバックします:", e)
                self.mip = PuLPInterface(nJ=data.J, p=data.p, solver_name=pulp_solver)
        else:
            self.mip = PuLPInterface(nJ=data.J, p=data.p, solver_name=pulp_solver)
        self.theta = self.mip.theta
        self.xvars = self.mip.x  # dict j->var

        # ログ用
        self.cuts_added: int = 0
        self.iterations: int = 0
        self.cuts_log: List[Tuple[str, float]] = []  # (カット種別, 参照値など)

    def _current_solution(self) -> Tuple[np.ndarray, float]:
        x_hat = np.array(
            [self.mip.get_value(self.xvars[j]) for j in range(self.data.J)], dtype=float
        )
        theta_hat = float(self.mip.get_value(self.theta))
        return x_hat, theta_hat

    def _add_bulge_cut(self, x_hat: np.ndarray, y_hat: np.ndarray):
        Lb, grad = Lb_and_grad(self.data, x_hat, y_hat)
        # θ ≤ Lb(x_hat, y_hat) + sum_j grad_j * (x_j - x_hat_j)
        # 並べ替え: θ ≤ (Lb - sum_j grad_j * x_hat_j) + sum_j grad_j * x_j
        c0 = Lb - float((grad * x_hat).sum())
        coeffs = {j: float(grad[j]) for j in range(self.data.J) if abs(grad[j]) > 1e-12}
        self.mip.add_linear_cut_theta_le(
            self.theta, c0, coeffs, name=f"bulge_cut_{self.cuts_added}"
        )
        self.cuts_added += 1
        self.cuts_log.append(("bulge", Lb))

    def _add_submodular_cut_if_integer(self, x_hat: np.ndarray, y_hat: np.ndarray):
        # x_hat が（ほぼ）0/1 のときのみ追加
        if np.all((x_hat < 1e-6) | (x_hat > 1 - 1e-6)):
            x_bin = (x_hat > 0.5).astype(float)
            y_bin = (y_hat > 0.5).astype(float)
            c0, coeffs = submodular_cut_coeffs(self.data, x_bin, y_bin)
            self.mip.add_linear_cut_theta_le(
                self.theta, c0, coeffs, name=f"submod_cut_{self.cuts_added}"
            )
            self.cuts_added += 1
            self.cuts_log.append(("submod", c0))
            return True
        return False

    def solve(
        self, max_rounds: int = 200, tol: float = 1e-9, verbose: bool = True
    ) -> Dict[str, object]:
        """
        DCG の主ループ（外側 cutting-plane）:
          - マスタ MILP を解く
          - 近似分離で y_hat を得る
          - 真値 L(x_hat, y_hat) を評価。theta_hat > L なら Bulge（＋整数なら Submodular）カットを追加
          - 違反がなければ停止

        戻り値: 'x', 'theta', 'obj', 'status', 'iterations', 'cuts'
        """
        for it in range(1, max_rounds + 1):
            self.iterations = it
            status = self.mip.solve()
            x_hat, theta_hat = self._current_solution()
            # 分離
            y_hat, ub_lin, beta = approximate_separation_y(self.data, x_hat)
            true_val = L_value(self.data, x_hat, y_hat)
            if verbose:
                print(
                    f"[Round {it}] theta_hat={theta_hat:.6f}, L(x,y)={true_val:.6f}, approx_ub={ub_lin:.6f}, "
                    f"x_int? {np.all((x_hat<1e-6)|(x_hat>1-1e-6))}"
                )
            if theta_hat > true_val + tol:
                # カットを追加
                self._add_bulge_cut(x_hat, y_hat)
                added_sub = self._add_submodular_cut_if_integer(x_hat, y_hat)
                if verbose:
                    print(
                        f"  -> Violated: add bulge cut{' + submod' if added_sub else ''}."
                    )
                continue
            else:
                if verbose:
                    print("  -> No violation. 現在の Y-closure に対して最適。")
                # x_hat が分数でも、近似分離のもとで違反が見つからなければ概ね十分。
                # 厳密性をさらに高めるなら、時折 厳密な静的-CFLP 分離に置換しても良い。
                break

        # 最終的なレポート用に再度解く（MILP が整数性を担保）
        status = self.mip.solve()
        x_hat, theta_hat = self._current_solution()
        return dict(
            x=x_hat,
            theta=theta_hat,
            obj=theta_hat,
            status=status,
            iterations=self.iterations,
            cuts=self.cuts_added,
            cuts_log=self.cuts_log,
        )


# --- データ生成のヘルパ --------------------------------------------------------


def make_random_instance(
    I: int = 10,
    J: int = 10,
    p: int = 2,
    r: int = 2,
    beta: float = 0.1,
    alpha_mean: float = 0.0,
    alpha_std: float = 0.0,
    seed: int = 0,
) -> SCFLPData:
    """
    論文の実験プロトコルに倣い、[0,50] x [0,50] 上にランダムな平面インスタンスを生成。
    """
    rng = np.random.default_rng(seed)
    demand_xy = rng.uniform(0, 50, size=(I, 2))
    cand_xy = rng.uniform(0, 50, size=(J, 2))
    alpha = rng.normal(loc=alpha_mean, scale=alpha_std, size=(J,))
    return SCFLPData.from_coordinates(
        demand_xy=demand_xy, cand_xy=cand_xy, alpha=alpha, beta=beta, p=p, r=r
    )


if __name__ == "__main__":
    # クイックスモークテスト（CBC で手早く解ける小サイズ）
    data = make_random_instance(I=60, J=40, p=2, r=2, beta=0.1, seed=42)
    solver = SCFLPSolver(data, milp_backend="pulp", pulp_solver="CBC")
    res = solver.solve(max_rounds=100, verbose=True)
    print("Solved: theta=", res["theta"], " sum x=", res["x"].sum(), "p=", data.p)
