# scflp_fn.py
# 逐次型 競争的施設配置問題 (S-CFLP) — 関数版（フレンドリーなログ）
# ・Branch-and-(outer)-Cut + 遅延制約生成（DCG）
# ・フォロワ側の分離は命題5の近似分離（単一ソート）
# ・追加カット: Bulge（常に） / Submodular（xが整数のとき）
# ・MILPはPuLP(CBC/GLPK)のみを使用（簡素化のため）

from __future__ import annotations

import time
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Set

# ========== データ生成・前処理（クラス廃止、dictで扱う） =========================


def normalize_h(h: np.ndarray) -> np.ndarray:
    s = float(h.sum())
    return h / s if abs(s - 1.0) > 1e-12 else h


def scflp_from_coordinates(
    demand_xy: np.ndarray,
    cand_xy: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    p: int,
    r: int,
    leader_xy: Optional[np.ndarray] = None,
    follower_xy: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """2次元座標からS-CFLPデータ(dict)を作る。"""
    I, J = demand_xy.shape[0], cand_xy.shape[0]
    h = normalize_h(np.ones(I) / I if h is None else h.astype(float))
    alpha = alpha.reshape(J)

    def pairwise_dist(A, B):
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

    d_cand = pairwise_dist(demand_xy, cand_xy)
    w = np.exp(alpha.reshape(1, J) - beta * d_cand)

    UL = np.zeros(I)
    UF = np.zeros(I)
    if leader_xy is not None and len(leader_xy) > 0:
        dL = pairwise_dist(demand_xy, leader_xy)
        wL = np.exp(alpha.mean() - beta * dL)
        UL = wL.sum(axis=1)
    if follower_xy is not None and len(follower_xy) > 0:
        dF = pairwise_dist(demand_xy, follower_xy)
        wF = np.exp(alpha.mean() - beta * dF)
        UF = wF.sum(axis=1)

    return dict(h=h, UL=UL, UF=UF, w=w, I=I, J=J, p=int(p), r=int(r))


def make_random_instance(
    I: int = 10,
    J: int = 10,
    p: int = 2,
    r: int = 2,
    beta: float = 0.1,
    alpha_mean: float = 0.0,
    alpha_std: float = 0.0,
    seed: int = 0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    demand_xy = rng.uniform(0, 50, size=(I, 2))
    cand_xy = rng.uniform(0, 50, size=(J, 2))
    alpha = rng.normal(loc=alpha_mean, scale=alpha_std, size=(J,))
    return scflp_from_coordinates(demand_xy, cand_xy, alpha, beta, p, r)


# ========== 目的関数・評価・分離・カット係数 ====================================


def L_value(data: Dict[str, object], x: np.ndarray, y: np.ndarray) -> float:
    """式(4)の真値 L(x,y)。"""
    w = data["w"]
    UL = data["UL"]
    UF = data["UF"]
    h = data["h"]
    x = x.reshape(-1)
    y = y.reshape(-1)
    xy = np.maximum(x, y)
    num = UL + w.dot(x)
    den = UL + UF + w.dot(xy)
    return float((h * (num / den)).sum())


def Lb_and_grad(
    data: Dict[str, object], x: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Bulge関数 L_b(x,y) と x勾配（式(11)の接平面）。"""
    w = data["w"]
    UL = data["UL"]
    UF = data["UF"]
    h = data["h"]
    J = w.shape[1]
    x = x.reshape(J)
    y = y.reshape(J)
    one_minus_y = 1.0 - y
    P = UL + UF + w.dot(one_minus_y * x + y)  # (I,)
    Q = UL + w.dot(-y * (x**2) + (1.0 + y) * x)  # (I,)
    Lb = (h * (Q / P)).sum()
    grad = np.zeros(J)
    P2 = P * P
    for j in range(J):
        wj = w[:, j]
        term = (
            -wj * one_minus_y[j] * Q / P2 + wj * (-2.0 * y[j] * x[j] + 1.0 + y[j]) / P
        )
        grad[j] = (h * term).sum()
    return float(Lb), grad


def submodular_cut_coeffs(
    data: Dict[str, object], x_bin: np.ndarray, y_bin: np.ndarray
) -> Tuple[float, Dict[int, float]]:
    """Nemhauser–Wolsey 線形化の係数（固定Y下、式(8)）。"""
    w = data["w"]
    UL = data["UL"]
    UF = data["UF"]
    h = data["h"]
    J = w.shape[1]
    S: Set[int] = set(np.where(x_bin > 0.5)[0].tolist())
    Y: Set[int] = set(np.where(y_bin > 0.5)[0].tolist())

    def L_Y_of_X(Xset: Set[int]) -> float:
        maskX = np.zeros(J)
        if Xset:
            maskX[list(Xset)] = 1.0
        maskXUY = maskX
        if Y:
            maskY = np.zeros(J)
            maskY[list(Y)] = 1.0
            maskXUY = np.maximum(maskX, maskY)
        num = UL + w.dot(maskX)
        den = UL + UF + w.dot(maskXUY)
        return float((h * (num / den)).sum())

    LYS = L_Y_of_X(S)

    rho_S_k: Dict[int, float] = {}
    for k in range(J):
        if k in S:
            continue
        Sk = set(S)
        Sk.add(k)
        rho_S_k[k] = L_Y_of_X(Sk) - LYS

    rho_Jminus_k_k: Dict[int, float] = {}
    Jall = set(range(J))
    for k in range(J):
        full_minus_k = set(Jall)
        full_minus_k.remove(k)
        LY_full_minus_k = L_Y_of_X(full_minus_k)
        rho_Jminus_k_k[k] = L_Y_of_X(Jall) - LY_full_minus_k

    c0 = LYS - sum(rho_Jminus_k_k[k] for k in S)
    coeffs: Dict[int, float] = {}
    for k in S:
        coeffs[k] = rho_Jminus_k_k[k]
    for k in range(J):
        if k not in S:
            coeffs[k] = rho_S_k[k]
    return c0, coeffs


def approximate_separation_y(
    data: Dict[str, object], x: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """命題5の近似分離。y_hat, ub_lin, beta を返す。"""
    w = data["w"]
    UL = data["UL"]
    UF = data["UF"]
    h = data["h"]
    r = data["r"]
    I, J = w.shape
    x = x.reshape(J)
    a = UL + w.dot(x)  # (I,)
    w_scaled = w * (1.0 - x.reshape(1, J))  # (I,J)

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
    wL = UF + smallest
    wU = UF + largest

    denomL = a + wL
    denomU = a + wU
    alpha_terms = a * (a + wU + wL - UF) / (denomU * denomL)
    alpha_val = float((h * alpha_terms).sum())
    common = h * (a / (denomU * denomL))
    beta = (common.reshape(I, 1) * (w * (1.0 - x.reshape(1, J)))).sum(axis=0)

    idx_sorted = np.argsort(-beta)
    y_hat = np.zeros(J)
    if r > 0:
        y_hat[idx_sorted[:r]] = 1.0
    ub_lin = alpha_val - float(beta[y_hat > 0.5].sum())
    return y_hat, ub_lin, beta


# ========== MILP（PuLP）ユーティリティ =========================================


def build_master_pulp(nJ: int, p: int, solver_name: str = "CBC"):
    import pulp  # lazy import

    model = pulp.LpProblem("S_CFLP_Master", pulp.LpMaximize)
    x = {
        j: pulp.LpVariable(f"x[{j}]", lowBound=0, upBound=1, cat="Binary")
        for j in range(nJ)
    }
    theta = pulp.LpVariable("theta", lowBound=0.0, upBound=1.0, cat="Continuous")
    # 予算制約と目的
    model += pulp.lpSum([x[j] for j in range(nJ)]) == p, "budget_eq"
    model += theta, "obj"
    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=False)
    elif solver_name == "GLPK":
        solver = pulp.GLPK_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)
    return model, x, theta, solver


def solve_master(model, solver) -> int:
    return int(model.solve(solver))


def get_values_pulp(xvars: Dict[int, object], theta) -> Tuple[np.ndarray, float]:
    x_hat = np.array([float(v.value()) for j, v in sorted(xvars.items())], dtype=float)
    return x_hat, float(theta.value())


def add_linear_cut_theta_le(
    model, theta, c0: float, coeffs: Dict[int, float], name: str
):
    """theta <= c0 + sum_j coeffs_j x_j  を  theta - sum <= c0 で追加"""
    expr = 0.0
    for j, cj in coeffs.items():
        expr += cj * model.variablesDict()[f"x[{j}]"]
    model += (theta - expr <= c0), name


# ========== ログ（関数版） =====================================================


def _term_width(default=100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def log_header(title: str, level: str):
    if level not in ("info", "debug"):
        return
    w = min(_term_width(), 80)
    print("═" * w)
    print(f"🚀 {title}")
    print("═" * w)


def log_line(level: str, char="─"):
    if level not in ("info", "debug"):
        return
    print(char * min(_term_width(), 80))


def log_step(msg: str, level: str):
    if level not in ("info", "debug"):
        return
    print(f"➤ {msg}")


def log_debug(msg: str, level: str):
    if level != "debug":
        return
    print(f"   · {msg}")


def log_cut(kinds: List[str], level: str):
    if level not in ("info", "debug"):
        return
    print(f"✂️  カット追加: {' + '.join(kinds).upper()}")


def log_success(msg: str, level: str):
    if level not in ("info", "debug"):
        return
    print(f"✅ {msg}")


def log_warn(msg: str, level: str):
    if level not in ("info", "debug"):
        return
    print(f"⚠️  {msg}")


def log_done(msg: str, tstart: float, level: str):
    if level not in ("info", "debug"):
        return
    w = min(_term_width(), 80)
    print(f"🏁 {msg}（経過 {time.time() - tstart:.2f}s）")
    print("═" * w)


# ========== メイン求解ルーチン（関数のみ） =====================================


def scflp_solve(
    data: Dict[str, object],
    max_rounds: int = 200,
    tol: float = 1e-8,
    pulp_solver: str = "CBC",
    log_level: str = "info",
) -> Dict[str, object]:
    """
    DCG 主ループ（outer cutting plane）:
      1) master MILP（θ最大化, Σx=p）を解く
      2) 近似分離で ŷ を構成（β降順にr個）
      3) 真の L(x̂,ŷ) を計算。違反があれば Bulge カット（＋整数なら Submodular）
      4) 違反なしで停止
    """
    J = int(data["J"])
    I = int(data["I"])
    p = int(data["p"])
    r = int(data["r"])

    # 準備
    model, xvars, theta, solver = build_master_pulp(J, p, solver_name=pulp_solver)
    cuts_added = 0
    cuts_log: List[Tuple[str, float]] = []
    t0 = time.time()

    bulge_count = 0
    submod_count = 0

    log_header("S-CFLP を解きます（DCG + Bulge/Submodular カット）", log_level)
    log_step(f"候補地 J = {J}, 需要点 I = {I}, p = {p}, r = {r}", log_level)

    for it in range(1, max_rounds + 1):
        it_start = time.time()
        status = solve_master(model, solver)
        x_hat, theta_hat = get_values_pulp(xvars, theta)
        is_int = bool(np.all((x_hat < 1e-6) | (x_hat > 1 - 1e-6)))

        log_line(log_level)
        log_step(
            f"🧮 ラウンド {it}: master 解 → θ̂ = {theta_hat:.6f}, |x| = {x_hat.sum():.0f}/{p}（{'整数' if is_int else '小数混在'}）",
            log_level,
        )

        # 近似分離
        y_hat, ub_lin, beta = approximate_separation_y(data, x_hat)
        y_idx = np.where(y_hat > 0.5)[0].tolist()
        true_val = L_value(data, x_hat, y_hat)
        gap = theta_hat - true_val
        log_step(
            f"🔎 分離：ŷ は {y_idx}（r = {r}）。上界 α − βᵀŷ = {ub_lin:.6f}", log_level
        )
        topk = np.argsort(-beta)[: min(5, J)]
        log_debug(
            "β 上位: " + ", ".join([f"j={int(j)}: β={beta[j]:.4g}" for j in topk]),
            log_level,
        )
        log_step(
            f"📐 評価：L(x̂,ŷ) = {true_val:.6f} → 違反量 θ̂ - L = {gap:.3e}", log_level
        )

        if theta_hat > true_val + tol:
            # Bulge cut
            Lb, grad = Lb_and_grad(data, x_hat, y_hat)
            c0 = Lb - float((grad * x_hat).sum())
            coeffs = {j: float(grad[j]) for j in range(J) if abs(grad[j]) > 1e-12}
            add_linear_cut_theta_le(
                model, theta, c0, coeffs, name=f"bulge_{cuts_added}"
            )
            cuts_added += 1
            bulge_count += 1
            cuts_log.append(("bulge", Lb))

            kinds = ["bulge"]
            if is_int:
                # Submodular cut
                x_bin = (x_hat > 0.5).astype(float)
                y_bin = (y_hat > 0.5).astype(float)
                c0_s, coeffs_s = submodular_cut_coeffs(data, x_bin, y_bin)
                add_linear_cut_theta_le(
                    model, theta, c0_s, coeffs_s, name=f"submod_{cuts_added}"
                )
                cuts_added += 1
                submod_count += 1
                cuts_log.append(("submod", c0_s))
                kinds.append("submod")

            log_cut(kinds, log_level)
            log_debug(
                f"このラウンド所要: {time.time()-it_start:.2f}s / 累計カット {cuts_added}",
                log_level,
            )
            continue
        else:
            log_success("違反なし。現在の Y-closure に対して最適です。", log_level)
            break

    # 最終化
    status = solve_master(model, solver)
    x_hat, theta_hat = get_values_pulp(xvars, theta)
    sel = np.where(x_hat > 0.5)[0].tolist()
    log_done("求解完了", t0, log_level)
    log_step(f"✨ 目的値 θ = {theta_hat:.6f} / 選択サイト {sel}", log_level)
    log_step(f"🧷 追加カット {cuts_added} 本", log_level)
    log_step(
        f"🧷 追加カット 合計 {cuts_added} 本（Bulge: {bulge_count}, Submodular: {submod_count}）",
        log_level,
    )
    if abs(x_hat.sum() - p) > 1e-6:
        log_warn("Σx が p と一致していません。予算制約を確認してください。", log_level)

    return dict(
        x=x_hat,
        theta=theta_hat,
        obj=theta_hat,
        status=status,
        iterations=it,
        cuts=cuts_added,
        cuts_log=cuts_log,
        bulge_cuts=bulge_count,
        submod_cuts=submod_count,
        selected_sites=sel,
        time_sec=time.time() - t0,
    )


# ========== 分枝付き Branch-and-Cut：ユーティリティ ============================


def build_master_with_fixings_and_cuts(
    nJ: int,
    p: int,
    fix1: Set[int],
    fix0: Set[int],
    cuts: List[Tuple[float, Dict[int, float]]],
    solver_name: str = "CBC",
):
    """
    分枝での固定と既存カットを含めて master を構築する。

    引数:
        nJ (int): 施設候補数 J
        p (int): 予算
        fix1 (Set[int]): x_j=1 に固定する添字集合
        fix0 (Set[int]): x_j=0 に固定する添字集合
        cuts (List[Tuple[c0, coeffs]]): 既に追加済みのカット（θ <= c0 + Σ coeffs_j x_j）
        solver_name (str): 'CBC' または 'GLPK'

    返り値:
        (model, xvars, theta, solver): PuLP のオブジェクト群
    """
    import pulp

    model = pulp.LpProblem("S_CFLP_Master", pulp.LpMaximize)
    xvars = {}
    for j in range(nJ):
        if j in fix1:
            xvars[j] = pulp.LpVariable(f"x[{j}]", lowBound=1, upBound=1, cat="Binary")
        elif j in fix0:
            xvars[j] = pulp.LpVariable(f"x[{j}]", lowBound=0, upBound=0, cat="Binary")
        else:
            xvars[j] = pulp.LpVariable(f"x[{j}]", lowBound=0, upBound=1, cat="Binary")

    theta = pulp.LpVariable("theta", lowBound=0.0, upBound=1.0, cat="Continuous")
    model += pulp.lpSum([xvars[j] for j in range(nJ)]) == p, "budget_eq"
    model += theta, "obj"

    # 既存カットを再追加
    for k, (c0, coeffs) in enumerate(cuts):
        expr = 0.0
        for j, cj in coeffs.items():
            expr += cj * xvars[j]
        model += (theta - expr <= c0), f"cut_replay_{k}"

    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=False)
    elif solver_name == "GLPK":
        solver = pulp.GLPK_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    return model, xvars, theta, solver


def choose_branch_variable(x_hat: np.ndarray) -> Optional[int]:
    """
    分枝で固定する変数 j* を選ぶ（0.5 に最も近い小数値）。

    引数:
        x_hat (np.ndarray): (J,) 現在の解

    返り値:
        int or None: 分枝対象の添字。すべて整数なら None。
    """
    frac = np.where((x_hat > 1e-6) & (x_hat < 1 - 1e-6))[0]
    if len(frac) == 0:
        return None
    j_star = int(frac[np.argmin(np.abs(x_hat[frac] - 0.5))])
    return j_star


def node_solve_until_no_violation(
    data: Dict[str, object],
    node: Dict[str, object],
    pulp_solver: str,
    tol: float,
    max_cuts_per_node: int = 200,
    log_level: str = "info",
):
    """
    1つのノード（部分問題）について、カット追加を繰り返して違反が無くなるまで解く。

    引数:
        data (dict): 問題データ
        node (dict): {"fix1": set, "fix0": set, "cuts": list[(c0, coeffs)], "depth": int}
        pulp_solver (str): PuLP ソルバ名
        tol (float): 収束判定許容
        max_cuts_per_node (int): 1ノードで追加可能なカットの上限
        log_level (str): 'info' or 'debug' or 'quiet'

    返り値:
        dict: {
          "status": int, "x_hat": np.ndarray, "theta_hat": float,
          "y_hat": np.ndarray, "true_val": float, "viol": float,
          "integral": bool, "cuts_added": int
        }
    """
    J = int(data["J"])
    total_new_cuts = 0
    bulge_added = 0
    submod_added = 0

    for _ in range(max_cuts_per_node):
        model, xvars, theta, solver = build_master_with_fixings_and_cuts(
            nJ=J,
            p=int(data["p"]),
            fix1=node["fix1"],
            fix0=node["fix0"],
            cuts=node["cuts"],
            solver_name=pulp_solver,
        )
        status = solve_master(model, solver)
        x_hat, theta_hat = get_values_pulp(xvars, theta)
        is_int = bool(np.all((x_hat < 1e-6) | (x_hat > 1 - 1e-6)))

        # 近似分離 & 真の評価
        y_hat, ub_lin, beta = approximate_separation_y(data, x_hat)
        true_val = L_value(data, x_hat, y_hat)
        gap = theta_hat - true_val

        log_step(
            f"   · ノード(depth={node['depth']}): θ̂={theta_hat:.6f}, L={true_val:.6f}, viol={gap:.3e}",
            log_level,
        )

        if theta_hat <= true_val + tol:
            # 違反なし → 収束
            return dict(
                status=status,
                x_hat=x_hat,
                theta_hat=theta_hat,
                y_hat=y_hat,
                true_val=true_val,
                viol=gap,
                integral=is_int,
                cuts_added=total_new_cuts,
                bulge_added=bulge_added,
                submod_added=submod_added,
            )

        # 違反あり → カット追加
        Lb, grad = Lb_and_grad(data, x_hat, y_hat)
        c0 = Lb - float((grad * x_hat).sum())
        coeffs = {j: float(grad[j]) for j in range(J) if abs(grad[j]) > 1e-12}
        node["cuts"].append((c0, coeffs))
        total_new_cuts += 1
        bulge_added += 1
        kinds = ["bulge"]

        if is_int:
            x_bin = (x_hat > 0.5).astype(float)
            y_bin = (y_hat > 0.5).astype(float)
            c0s, coeffs_s = submodular_cut_coeffs(data, x_bin, y_bin)
            node["cuts"].append((c0s, coeffs_s))
            total_new_cuts += 1
            submod_added += 1
            kinds.append("submod")

        log_cut(kinds, log_level)

        if total_new_cuts >= max_cuts_per_node:
            log_warn(
                "このノードでのカット上限に到達。中断して上位で処理します。", log_level
            )
            break

    # ループを抜けた場合（上限到達など）
    return dict(
        status=status,
        x_hat=x_hat,
        theta_hat=theta_hat,
        y_hat=y_hat,
        true_val=true_val,
        viol=gap,
        integral=is_int,
        cuts_added=total_new_cuts,
        bulge_added=bulge_added,
        submod_added=submod_added,
    )


def make_node(
    fix1: Optional[Set[int]] = None,
    fix0: Optional[Set[int]] = None,
    cuts: Optional[List[Tuple[float, Dict[int, float]]]] = None,
    depth: int = 0,
):
    """
    分枝ノード（部分問題）を作る簡易ファクトリ。

    引数:
        fix1 (set[int] | None): x_j=1 に固定する添字集合
        fix0 (set[int] | None): x_j=0 に固定する添字集合
        cuts (list | None): 既存カット（(c0, coeffs) のリスト）
        depth (int): 探索木の深さ

    返り値:
        dict: {"fix1", "fix0", "cuts", "depth", "ub"}
    """
    return dict(
        fix1=set() if fix1 is None else set(fix1),
        fix0=set() if fix0 is None else set(fix0),
        cuts=[] if cuts is None else list(cuts),
        depth=int(depth),
        ub=float("inf"),  # 上界（直近 θ̂）
    )


# ========== メイン：完全 Branch-and-Cut（分枝含む） =============================


def scflp_branch_and_cut(
    data: Dict[str, object],
    max_nodes: int = 10_000,
    max_rounds_per_node: int = 200,
    tol: float = 1e-8,
    pulp_solver: str = "CBC",
    node_selection: str = "dfs",  # 'dfs' or 'bestbound'
    log_level: str = "info",
) -> Dict[str, object]:
    """
    S-CFLP の完全 Branch-and-Cut（分枝 + カット生成）を解く。

    アルゴリズム（Algorithm 1 に対応）:
        1) ルートノードに緩和問題を入れて初期化
        2) ノードを取り出す（DFS 既定）
        3) ノード内で cutting-plane を収束させる（違反がなくなるまで）
        4) θ̂ <= θ_LB なら枝刈り
           4-1) x が整数ならインカンベント更新
           4-2) x が小数なら最も 0.5 に近い x_j で分枝して 2 子ノード生成
        5) ノード集合が空になるまで繰り返す

    引数:
        data (dict): 問題データ辞書
        max_nodes (int): ノード探索総数の上限
        max_rounds_per_node (int): 1ノード内のカット反復上限
        tol (float): 収束判定用の許容
        pulp_solver (str): PuLP ソルバー名（CBC/GLPK）
        node_selection (str): 'dfs'（stack）または 'bestbound'（最大上界選択）
        log_level (str): 'info' / 'debug' / 'quiet'

    返り値:
        dict: {
          "x_best": np.ndarray or None,
          "theta_best": float,
          "status": str,
          "nodes_explored": int,
          "time_sec": float,
          "gap_bound": float  # (上界-下界)
        }
    """
    J = int(data["J"])
    p = int(data["p"])
    t0 = time.time()

    # 下界・ベスト解
    theta_LB = 0.0
    x_best = None

    total_bulge = 0
    total_submod = 0
    total_cuts = 0

    # ルートノード
    root = make_node()
    if p < 0 or p > J:
        return dict(
            x_best=None,
            theta_best=theta_LB,
            status="infeasible",
            nodes_explored=0,
            time_sec=0.0,
            gap_bound=float("inf"),
        )

    # ノード集合
    nodes: List[Dict[str, object]] = [root]

    explored = 0
    log_header("Branch-and-Cut（分枝＋カット）開始", log_level)

    while nodes and explored < max_nodes:
        # ノード取り出し
        if node_selection == "bestbound":
            # ub が最大のもの
            idx = int(np.argmax([n.get("ub", -1e100) for n in nodes]))
            node = nodes.pop(idx)
        else:
            # DFS
            node = nodes.pop()

        explored += 1
        log_line(log_level)
        log_step(
            f"🌿 ノード展開: depth={node['depth']}, |fix1|={len(node['fix1'])}, |fix0|={len(node['fix0'])}",
            log_level,
        )

        # 予算の早期チェック：すでに x=1 固定が p 超えなら infeasible
        if len(node["fix1"]) > p:
            log_warn("早期枝刈り: |fix1| > p", log_level)
            continue

        # ノード内で cutting-plane を収束させる
        info = node_solve_until_no_violation(
            data=data,
            node=node,
            pulp_solver=pulp_solver,
            tol=tol,
            max_cuts_per_node=max_rounds_per_node,
            log_level=log_level,
        )
        theta_hat = info["theta_hat"]
        x_hat = info["x_hat"]
        is_int = info["integral"]
        node["ub"] = float(theta_hat)

        total_cuts += int(info.get("cuts_added", 0))
        total_bulge += int(info.get("bulge_added", 0))
        total_submod += int(info.get("submod_added", 0))

        # 上界による枝刈り
        if theta_hat <= theta_LB + tol:
            log_step(f"🪓 枝刈り: θ̂={theta_hat:.6f} <= θ_LB={theta_LB:.6f}", log_level)
            continue

        # 整数解ならインカンベント更新
        if is_int:
            x_bin = (x_hat > 0.5).astype(float)
            theta_LB = float(theta_hat)
            x_best = x_bin
            log_success(f"🎯 インカンベント更新: θ_LB ← {theta_LB:.6f}", log_level)
            continue

        # 分枝（最も 0.5 に近い変数）
        j_star = choose_branch_variable(x_hat)
        if j_star is None:
            # 小数判定に来ないはずだが保険
            x_bin = (x_hat > 0.5).astype(float)
            theta_LB = max(theta_LB, float(theta_hat))
            x_best = x_bin
            log_warn("x は実質整数だったため更新のみ。", log_level)
            continue

        print("分岐するぞ")
        # 子ノード 2 個（x_j=0, x_j=1）
        child0 = make_node(
            fix1=node["fix1"],
            fix0=node["fix0"] | {j_star},
            cuts=node["cuts"],
            depth=node["depth"] + 1,
        )

        child1 = make_node(
            fix1=node["fix1"] | {j_star},
            fix0=node["fix0"],
            cuts=node["cuts"],
            depth=node["depth"] + 1,
        )

        # 早期不可能性: child1 の固定が p を超えたら捨てる
        if len(child1["fix1"]) > p:
            log_warn(f"子ノード(x[{j_star}]=1)は |fix1|>p のため破棄", log_level)
        else:
            nodes.append(child1)

        # child0 は x_j=0 なので常に可
        nodes.append(child0)
        log_step(
            f"🌱 分枝: j*={j_star} → 子ノード depth={node['depth']+1} を2つ追加",
            log_level,
        )

    # 終了
    elapsed = time.time() - t0
    gap = (
        (max([n.get("ub", 0.0) for n in nodes], default=theta_LB) - theta_LB)
        if nodes
        else 0.0
    )
    status = "optimal" if not nodes else "stopped_or_pruned"

    log_done("Branch-and-Cut 終了", t0, log_level)
    if x_best is not None:
        sel = np.where(x_best > 0.5)[0].tolist()
        log_step(f"✨ 最良解 θ = {theta_LB:.6f} / 選択サイト {sel}", log_level)
    else:
        log_warn("実行中にインカンベントは見つかりませんでした。", log_level)

    log_step(
        f"🧾 カット総計: {total_cuts} 本（Bulge: {total_bulge}, Submodular: {total_submod}）",
        log_level,
    )

    return dict(
        x_best=x_best,
        theta_best=float(theta_LB),
        status=status,
        nodes_explored=explored,
        time_sec=elapsed,
        gap_bound=float(gap),
        total_cuts=total_cuts,
        total_bulge=total_bulge,
        total_submod=total_submod,
    )
