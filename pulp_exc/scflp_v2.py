# scflp.py
# 逐次型 競争的施設配置問題 (S-CFLP) — Python 実装（フレンドリーなログ版）
# Branch-and-(outer)-Cut と遅延制約生成 (DCG)
# - 目的関数とモデルは Qi, Jiang, Shen (2022), arXiv:2103.04259v3 に準拠
# - フォロワ側の分離は命題5の近似分離（単一ソート）を使用
# - 2種類のカットを追加: Submodular（整数 x のとき）と Bulge（常に）
#
# 依存関係:
#   numpy（必須）
#   PuLP（推奨のフォールバック）または gurobipy（任意、MILP を高速に解く場合）
#
# 変更点（ログ強化）:
# - 親しみやすい絵文字＆日本語で進捗を表示（各ラウンドの概要、分離の要約、追加したカット等）
# - ログレベル: 'quiet' / 'info' / 'debug'
# - 便利なまとめ出力（総時間・反復数・追加カット数・選択サイト・最終ギャップなど）

from __future__ import annotations

import time
import math
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Sequence, Set

# --- オプションの MILP バックエンド -------------------------------------------
# 最小限の抽象化により、PuLP か Gurobi のどちらでも差し替え可能。
# 既定: PuLP (CBC)。gurobipy が利用可能なら 'gurobi' を選択可。


class _MILPVar:
    def __init__(self, name, index=None):
        self.name = f"{name}[{index}]" if index is not None else name


class _MILPInterface:
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
            self.solver = pulp.PULP_CBC_CMD(msg=False)

    def add_binary_var(self, name: str, index: Optional[int] = None):
        raise NotImplementedError("Binary x_j created in constructor.")

    def add_continuous_var(self, name: str, lb: float, ub: float):
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
    UL : (I,) array           # 既存リーダー施設の効用
    UF : (I,) array           # 既存フォロワ施設の効用
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

        def pairwise_dist(A, B):
            return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

        d_cand = pairwise_dist(demand_xy, cand_xy)  # (I,J)
        w = np.exp(alpha.reshape(1, J) - beta * d_cand)

        UL = np.zeros(I)
        UF = np.zeros(I)
        if leader_xy is not None and leader_xy.shape[0] > 0:
            dL = pairwise_dist(demand_xy, leader_xy)
            wL = np.exp(alpha.mean() - beta * dL)  # 既存にも平均 α を仮定
            UL = wL.sum(axis=1)
        if follower_xy is not None and follower_xy.shape[0] > 0:
            dF = pairwise_dist(demand_xy, follower_xy)
            wF = np.exp(alpha.mean() - beta * dF)
            UF = wF.sum(axis=1)

        return SCFLPData(h=h, UL=UL, UF=UF, w=w, p=p, r=r)


# --- ユーティリティ関数 --------------------------------------------------------


def L_value(data: SCFLPData, x: np.ndarray, y: np.ndarray) -> float:
    """
    式(4)の L(x, y) を計算:  θ ≤ L(x,y) で使う真の評価値
      L(x,y) = sum_i h_i * (UL_i + sum_j w_ij x_j) / (UL_i + UF_i + sum_j w_ij (x_j OR y_j))
    """
    J = data.J
    x = x.reshape(J)
    y = y.reshape(J)
    xy = np.maximum(x, y)  # x ∨ y
    num = data.UL + data.w.dot(x)
    den = data.UL + data.UF + data.w.dot(xy)
    return float((data.h * (num / den)).sum())


def Lb_and_grad(
    data: SCFLPData, x: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Bulge 関数 L_b(x, y) と、その点 (x,y) における x に関する勾配。
    （式(11) の接平面を切断として使う）
    """
    J = data.J
    x = x.reshape(J)
    y = y.reshape(J)
    one_minus_y = 1.0 - y
    P = data.UL + data.UF + data.w.dot(one_minus_y * x + y)  # (I,)
    Q = data.UL + data.w.dot(-y * (x**2) + (1.0 + y) * x)  # (I,)
    Lb = (data.h * (Q / P)).sum()
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
    固定された Y の下で、S = supp(x) を用いて Nemhauser–Wolsey 線形化を構築（式(8)）。
    返り値 (c0, coeffs) は不等式 θ <= c0 + sum_j coeffs[j] * x_j の係数。
    """
    J = data.J
    S: Set[int] = set(np.where(x_bin > 0.5)[0].tolist())
    Y: Set[int] = set(np.where(y_bin > 0.5)[0].tolist())

    def L_Y_of_X(Xset: Set[int]) -> float:
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

    LYS = L_Y_of_X(S)
    rho_S_k: Dict[int, float] = {}
    for k in range(J):
        if k in S:
            continue
        Sk = set(S)
        Sk.add(k)
        rho_S_k[k] = L_Y_of_X(Sk) - LYS

    rho_Jminus_k_k: Dict[int, float] = {}
    J_minus_k_base = set(range(J))
    for k in range(J):
        Sfull_minus_k = set(J_minus_k_base)
        Sfull_minus_k.remove(k)
        LY_full_minus_k = L_Y_of_X(Sfull_minus_k)
        Sfull = set(J_minus_k_base)
        rho_Jminus_k_k[k] = L_Y_of_X(Sfull) - LY_full_minus_k

    c0 = LYS - sum(rho_Jminus_k_k[k] for k in S)
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
    命題5の近似分離（p.16–17）。
      - α(x) と β(x) を閉形式で計算
      - y_hat は β(x) 降順に r 個
      - 戻り値: (y_hat, ub_lin = α(x) - β^T y_hat, beta)
    """
    J = data.J
    x = x.reshape(J)
    I = data.I
    a = data.UL + data.w.dot(x)  # (I,)
    w_scaled = data.w * (1.0 - x.reshape(1, J))  # (I,J)
    r = data.r
    smallest = (
        np.partition(w_scaled, r - 1, axis=1)[:, :r].sum(axis=1)
        if r > 0
        else np.zeros(I)
    )
    largest = (
        (-np.partition(-w_scaled, r - 1, axis=1)[:, :r]).sum(axis=1)
        if r > 0
        else np.zeros(I)
    )
    wL = data.UF + smallest
    wU = data.UF + largest

    denomL = a + wL
    denomU = a + wU
    alpha_terms = a * (a + wU + wL - data.UF) / (denomU * denomL)
    alpha_val = float((data.h * alpha_terms).sum())

    common = data.h * (a / (denomU * denomL))  # (I,)
    beta = (common.reshape(I, 1) * (data.w * (1.0 - x.reshape(1, J)))).sum(
        axis=0
    )  # (J,)

    idx_sorted = np.argsort(-beta)
    y_hat = np.zeros(J)
    if r > 0:
        y_hat[idx_sorted[:r]] = 1.0
    ub_lin = alpha_val - float(beta[y_hat > 0.5].sum())
    return y_hat, ub_lin, beta


# --- フレンドリー・ロガー ------------------------------------------------------


class FriendlyLogger:
    def __init__(
        self, level: str = "info", enabled: bool = True, width: Optional[int] = None
    ):
        self.level = level.lower()
        self.enabled = enabled
        self.start = time.time()
        try:
            self.width = width or shutil.get_terminal_size((100, 20)).columns
        except Exception:
            self.width = 100

    def _ok(self) -> bool:
        return self.enabled and self.level in ("info", "debug")

    def _dbg(self) -> bool:
        return self.enabled and self.level == "debug"

    def line(self, char: str = "─"):
        if not self._ok():
            return
        print(char * min(self.width, 80))

    def header(self, title: str):
        if not self._ok():
            return
        self.line("═")
        print(f"🚀 {title}")
        self.line("═")

    def step(self, msg: str):
        if not self._ok():
            return
        print(f"➤ {msg}")

    def debug(self, msg: str):
        if not self._dbg():
            return
        print(f"   · {msg}")

    def cut(self, kinds: List[str]):
        if not self._ok():
            return
        tag = " + ".join(kinds).upper()
        print(f"✂️  カット追加: {tag}")

    def success(self, msg: str):
        if not self._ok():
            return
        print(f"✅ {msg}")

    def warn(self, msg: str):
        if not self._ok():
            return
        print(f"⚠️  {msg}")

    def done(self, msg: str):
        if not self._ok():
            return
        elapsed = time.time() - self.start
        print(f"🏁 {msg}（経過 {elapsed:.2f}s）")
        self.line("═")


# --- メインソルバ --------------------------------------------------------------


class SCFLPSolver:
    """
    近似分離（命題5）と 2 種のカット（Submodular, Bulge）を用いた DCG により S-CFLP を解く。
    gurobipy が利用可能なら milp_backend='gurobi' を推奨。既定は PuLP (CBC)。
    """

    def __init__(
        self,
        data: SCFLPData,
        milp_backend: str = "pulp",
        pulp_solver: str = "CBC",
        log_level: str = "info",
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

        # ログ
        self.logger = FriendlyLogger(level=log_level, enabled=True)

        # ログ情報
        self.cuts_added: int = 0
        self.iterations: int = 0
        self.cuts_log: List[Tuple[str, float]] = []  # (種別, 値)

    # 内部ヘルパ
    def _current_solution(self) -> Tuple[np.ndarray, float]:
        x_hat = np.array(
            [self.mip.get_value(self.xvars[j]) for j in range(self.data.J)], dtype=float
        )
        theta_hat = float(self.mip.get_value(self.theta))
        return x_hat, theta_hat

    def _add_bulge_cut(self, x_hat: np.ndarray, y_hat: np.ndarray):
        Lb, grad = Lb_and_grad(self.data, x_hat, y_hat)
        c0 = Lb - float((grad * x_hat).sum())
        coeffs = {j: float(grad[j]) for j in range(self.data.J) if abs(grad[j]) > 1e-12}
        self.mip.add_linear_cut_theta_le(
            self.theta, c0, coeffs, name=f"bulge_cut_{self.cuts_added}"
        )
        self.cuts_added += 1
        self.cuts_log.append(("bulge", Lb))

    def _add_submodular_cut_if_integer(
        self, x_hat: np.ndarray, y_hat: np.ndarray
    ) -> bool:
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

    def _format_top(self, beta: np.ndarray, k: int = 5) -> str:
        k = min(k, len(beta))
        idx = np.argsort(-beta)[:k]
        pairs = [f"j={int(j)}: β={beta[j]:.4g}" for j in idx]
        return ", ".join(pairs)

    def solve(
        self, max_rounds: int = 200, tol: float = 1e-8, verbose: bool = True
    ) -> Dict[str, object]:
        """
        DCG の主ループ（外側 cutting-plane）:
          1) マスタ MILP を解く（θ 最大化, Σx=p）
          2) 近似分離で y_hat を取得（β 降順に r 個）
          3) 真の L(x_hat,y_hat) を評価。違反なら Bulge（＋整数なら Submodular）を追加
          4) 違反なしで停止
        """
        self.logger.header("S-CFLP を解きます（DCG + Bulge/Submodular カット）")
        self.logger.step(
            f"候補地 J = {self.data.J}, 需要点 I = {self.data.I}, p = {self.data.p}, r = {self.data.r}"
        )

        t0 = time.time()
        for it in range(1, max_rounds + 1):
            self.iterations = it
            it_start = time.time()
            status = self.mip.solve()
            x_hat, theta_hat = self._current_solution()
            is_int = bool(np.all((x_hat < 1e-6) | (x_hat > 1 - 1e-6)))
            self.logger.line()
            self.logger.step(
                f"🧮 ラウンド {it}: master 解 → \hat(θ) = {theta_hat:.6f}, "
                f"|x| = {x_hat.sum():.0f}/{self.data.p}（{'整数' if is_int else '小数混在'}）"
            )

            # 分離（命題5）
            y_hat, ub_lin, beta = approximate_separation_y(self.data, x_hat)
            y_idx = np.where(y_hat > 0.5)[0].tolist()
            true_val = L_value(self.data, x_hat, y_hat)
            gap = theta_hat - true_val
            self.logger.step(
                f"🔎 分離：ŷ は {y_idx}（r = {self.data.r}）。上界 α − βᵀŷ = {ub_lin:.6f}"
            )
            self.logger.debug(
                f"β 上位: {self._format_top(beta, k=min(5, self.data.J))}"
            )
            self.logger.step(
                f"📐 評価：L(x̂,ŷ) = {true_val:.6f} → 違反量 \hat(θ) - L = {gap:.3e}"
            )

            if theta_hat > true_val + tol:
                kinds = ["bulge"]
                self._add_bulge_cut(x_hat, y_hat)
                added_sub = self._add_submodular_cut_if_integer(x_hat, y_hat)
                if added_sub:
                    kinds.append("submod")
                self.logger.cut(kinds)
                self.logger.debug(f"現在のカット総数: {self.cuts_added}")
                self.logger.debug(f"このラウンド所要: {time.time()-it_start:.2f}s")
                continue
            else:
                self.logger.success("違反なし。現在の Y-closure に対して最適です。")
                break

        # 最終レポート
        status = self.mip.solve()
        x_hat, theta_hat = self._current_solution()
        elapsed = time.time() - t0
        sel = np.where(x_hat > 0.5)[0].tolist()

        self.logger.done("求解完了")
        self.logger.step(f"✨ 目的値 θ = {theta_hat:.6f} / 選択サイト {sel}")
        self.logger.step(
            f"🧷 反復 {self.iterations} 回・追加カット {self.cuts_added} 本・合計時間 {elapsed:.2f}s"
        )
        if abs(x_hat.sum() - self.data.p) > 1e-6:
            self.logger.warn("Σx が p と一致していません。予算制約を確認してください。")
        return dict(
            x=x_hat,
            theta=theta_hat,
            obj=theta_hat,
            status=status,
            iterations=self.iterations,
            cuts=self.cuts_added,
            cuts_log=self.cuts_log,
            selected_sites=sel,
            time_sec=elapsed,
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
    [0,50] x [0,50] 上にランダムな平面インスタンスを生成（論文の実験プロトコル）。
    """
    rng = np.random.default_rng(seed)
    demand_xy = rng.uniform(0, 50, size=(I, 2))
    cand_xy = rng.uniform(0, 50, size=(J, 2))
    alpha = rng.normal(loc=alpha_mean, scale=alpha_std, size=(J,))
    return SCFLPData.from_coordinates(
        demand_xy=demand_xy, cand_xy=cand_xy, alpha=alpha, beta=beta, p=p, r=r
    )


if __name__ == "__main__":
    data = make_random_instance(I=40, J=30, p=3, r=2, beta=0.1, seed=42)
    solver = SCFLPSolver(data, milp_backend="pulp", pulp_solver="CBC", log_level="info")
    res = solver.solve(max_rounds=200, verbose=True)
    print("\nSolved: theta=", res["theta"], " sum x=", res["x"].sum(), "p=", data.p)
