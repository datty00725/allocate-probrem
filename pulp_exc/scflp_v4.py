# scflp_fn.py
# 逐次型 競争的施設配置問題 (S-CFLP) — 関数版（論文 Algorithm 1 に忠実）
# ・Branch-and-(outer)-Cut + 遅延制約生成（DCG）
# ・分離は「近似」(命題5) または「厳密」を選択可（既定: 近似）
# ・カットは 1 回の違反につき 1 本のみ追加（Bulge か Submodular）
# ・MILP は PuLP(CBC/GLPK)

from __future__ import annotations

import time
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Set

# ========== データ生成・前処理 ==================================================


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


def build_master_pulp(nJ, p, solver_name="CBC", lp_relax=True):
    import pulp

    model = pulp.LpProblem("S_CFLP_Master", pulp.LpMaximize)
    cat = "Continuous" if lp_relax else "Binary"
    x = {
        j: pulp.LpVariable(f"x[{j}]", lowBound=0, upBound=1, cat=cat) for j in range(nJ)
    }
    theta = pulp.LpVariable("theta", lowBound=0.0, upBound=1.0, cat="Continuous")
    model += pulp.lpSum([x[j] for j in range(nJ)]) == p, "budget_eq"
    model += theta
    solver = (
        pulp.PULP_CBC_CMD(msg=False)
        if solver_name == "CBC"
        else pulp.GLPK_CMD(msg=False)
    )
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


# ========== ログ ===============================================================


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


def log_cut(kind: str, level: str):
    if level not in ("info", "debug"):
        return
    print(f"✂️  カット追加: {kind.upper()}")


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


# ========== Outer Cutting-Plane（論文 行1,5–9 部分） ===========================


def scflp_solve(
    data: Dict[str, object],
    max_rounds: int = 200,
    tol: float = 1e-12,
    pulp_solver: str = "CBC",
    log_level: str = "info",
    separation: str = "approx",  # "approx" or "exact"（exactは未実装）
    cut_policy: str = "auto",  # "bulge" / "submodular" / "auto"
) -> Dict[str, object]:
    """
    論文の外側 cutting-plane（分枝なし）。違反がある限り、1 回に 1 本のカットを追加。
    """
    J = int(data["J"])
    I = int(data["I"])
    p = int(data["p"])
    r = int(data["r"])

    model, xvars, theta, solver = build_master_pulp(J, p, solver_name=pulp_solver)
    cuts_added = 0
    cuts_log: List[Tuple[str, float]] = []
    t0 = time.time()
    bulge_count = 0
    submod_count = 0

    log_header("S-CFLP を解きます（Outer Cutting-Plane）", log_level)
    log_step(f"候補地 J = {J}, 需要点 I = {I}, p = {p}, r = {r}", log_level)

    for it in range(1, max_rounds + 1):
        log_line(log_level)
        status = solve_master(model, solver)
        x_hat, theta_hat = get_values_pulp(xvars, theta)
        is_int = bool(np.all((x_hat < 1e-6) | (x_hat > 1 - 1e-6)))
        log_step(
            f"🧮 ラウンド {it}: θ̂ = {theta_hat:.6f}, |x| = {x_hat.sum():.0f}/{p}（{'整数' if is_int else '小数混在'}）",
            log_level,
        )

        # 分離（既定: 近似）
        if separation == "approx":
            y_hat, ub_lin, beta = approximate_separation_y(data, x_hat)
        else:
            # exact 分離(6)が未実装の場合も API を保つ
            y_hat, ub_lin, beta = approximate_separation_y(data, x_hat)

        true_val = L_value(data, x_hat, y_hat)
        gap = theta_hat - true_val
        y_idx = np.where(y_hat > 0.5)[0].tolist()
        log_step(f"🔎 分離: ŷ = {y_idx}（r={r}）, 上界 α−βᵀŷ = {ub_lin:.6f}", log_level)
        log_step(f"📐 評価: L(x̂,ŷ) = {true_val:.6f} → 違反 {gap:.3e}", log_level)

        # 論文 行7: 違反ありなら 1 本だけカット追加
        if theta_hat > true_val + tol:
            kind_used = None

            # カット選択
            policy = cut_policy.lower()
            if policy == "submodular" or (policy == "auto" and is_int):
                # x が整数なら Submodular（auto でもここに来る）
                x_bin = (x_hat > 0.5).astype(float)
                y_bin = (y_hat > 0.5).astype(float)
                c0, coeffs = submodular_cut_coeffs(data, x_bin, y_bin)
                add_linear_cut_theta_le(
                    model, theta, c0, coeffs, f"submod_{cuts_added}"
                )
                kind_used = "submod"
                submod_count += 1
            elif policy == "bulge" or (policy == "auto" and not is_int):
                # 分数の間は Bulge（auto でもここ）
                Lb, grad = Lb_and_grad(data, x_hat, y_hat)
                c0 = Lb - float((grad * x_hat).sum())
                coeffs = {j: float(grad[j]) for j in range(J) if abs(grad[j]) > 1e-12}
                add_linear_cut_theta_le(model, theta, c0, coeffs, f"bulge_{cuts_added}")
                kind_used = "bulge"
                bulge_count += 1
            else:
                raise ValueError(f"unknown cut_policy: {policy}")

            cuts_added += 1
            cuts_log.append((kind_used, c0))
            log_cut(kind_used, log_level)
            continue  # 強化した同一フォーミュレーションを再解く（行9）
        else:
            log_success("違反なし。現在の Y-closure に対して最適です。", log_level)
            break

    # 仕上げ
    status = solve_master(model, solver)
    x_hat, theta_hat = get_values_pulp(xvars, theta)
    sel = np.where(x_hat > 0.5)[0].tolist()
    log_done("求解完了", t0, log_level)
    log_step(f"✨ 目的値 θ = {theta_hat:.6f} / 選択サイト {sel}", log_level)
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
    local_cuts: List[Tuple[float, Dict[int, float]]],
    global_cuts: List[Tuple[float, Dict[int, float]]],
    solver_name: str = "CBC",
    lp_relax: bool = True,
):
    """
    分枝での固定と既存カットを含めて master を構築する。
    論文 行8: 「全フォーミュレーション」に効くよう、global_cuts も投入。
    """
    import pulp

    model = pulp.LpProblem("S_CFLP_Master", pulp.LpMaximize)
    cat = "Continuous" if lp_relax else "Binary"
    xvars = {}
    for j in range(nJ):
        if j in fix1:
            xvars[j] = pulp.LpVariable(
                f"x[{j}]", lowBound=1, upBound=1, cat="Continuous"
            )  # 固定
        elif j in fix0:
            xvars[j] = pulp.LpVariable(
                f"x[{j}]", lowBound=0, upBound=0, cat="Continuous"
            )  # 固定
        else:
            xvars[j] = pulp.LpVariable(f"x[{j}]", lowBound=0, upBound=1, cat=cat)
    theta = pulp.LpVariable("theta", lowBound=0.0, upBound=1.0, cat="Continuous")
    model += pulp.lpSum([xvars[j] for j in range(nJ)]) == p, "budget_eq"
    model += theta

    # 既存カット（local + global）を再追加
    def replay(mdl, cuts, prefix):
        for k, (c0, coeffs) in enumerate(cuts):
            expr = 0.0
            for j, cj in coeffs.items():
                expr += cj * xvars[j]
            mdl += (theta - expr <= c0), f"{prefix}_{k}"

    replay(model, global_cuts, "gcut")
    replay(model, local_cuts, "lcut")

    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=False)
    elif solver_name == "GLPK":
        solver = pulp.GLPK_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    return model, xvars, theta, solver


def choose_branch_variable(x_hat: np.ndarray) -> Optional[int]:
    """最も 0.5 に近い分数変数で分枝。"""
    frac = np.where((x_hat > 1e-6) & (x_hat < 1 - 1e-6))[0]
    if len(frac) == 0:
        return None
    j_star = int(frac[np.argmin(np.abs(x_hat[frac] - 0.5))])
    return j_star


def make_node(
    fix1: Optional[Set[int]] = None,
    fix0: Optional[Set[int]] = None,
    cuts: Optional[List[Tuple[float, Dict[int, float]]]] = None,
    depth: int = 0,
):
    """分枝ノード（部分問題）を作る簡易ファクトリ。"""
    return dict(
        fix1=set() if fix1 is None else set(fix1),
        fix0=set() if fix0 is None else set(fix0),
        cuts=[] if cuts is None else list(cuts),
        depth=int(depth),
        ub=float("-inf"),  # 直近 θ̂
    )


# ========== 1ノード内の cutting-plane（違反が無くなるまで） ====================


def node_solve_until_no_violation(
    data: Dict[str, object],
    node: Dict[str, object],
    global_cuts: List[Tuple[float, Dict[int, float]]],
    pulp_solver: str,
    tol: float,
    cut_policy: str = "auto",  # "bulge"/"submodular"/"auto"
    separation: str = "approx",  # "approx"/"exact"
    max_cuts_per_node: int = 200,
    log_level: str = "info",
    lp_relax: bool = True,
):
    """
    論文 行5–9: 1ノードのフォーミュレーションを
    違反がなくなるまで強化して解く（毎回カットは 1 本のみ追加）。
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
            local_cuts=node["cuts"],
            global_cuts=global_cuts,
            solver_name=pulp_solver,
            lp_relax=lp_relax,
        )
        status = solve_master(model, solver)
        x_hat, theta_hat = get_values_pulp(xvars, theta)
        is_int = bool(np.all((x_hat < 1e-6) | (x_hat > 1 - 1e-6)))

        # 分離
        if separation == "approx":
            y_hat, _, _ = approximate_separation_y(data, x_hat)
        else:
            y_hat, _, _ = approximate_separation_y(data, x_hat)
        true_val = L_value(data, x_hat, y_hat)
        gap = theta_hat - true_val

        log_step(
            f"   · ノード(depth={node['depth']}): θ̂={theta_hat:.6f}, L={true_val:.6f}, viol={gap:.3e}",
            log_level,
        )

        # 違反なし → 終了（行10以降は上位で分岐判断）
        if theta_hat <= true_val + tol:
            node["ub"] = float(theta_hat)
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

        # 違反あり → カット 1 本追加（行8）
        policy = cut_policy.lower()
        if policy == "submodular" or (policy == "auto" and is_int):
            x_bin = (x_hat > 0.5).astype(float)
            y_bin = (y_hat > 0.5).astype(float)
            c0, coeffs = submodular_cut_coeffs(data, x_bin, y_bin)
            node["cuts"].append((c0, coeffs))
            global_cuts.append((c0, coeffs))
            submod_added += 1
            added_kind = "submod"
        elif policy == "bulge" or (policy == "auto" and not is_int):
            Lb, grad = Lb_and_grad(data, x_hat, y_hat)
            c0 = Lb - float((grad * x_hat).sum())
            coeffs = {j: float(grad[j]) for j in range(J) if abs(grad[j]) > 1e-12}
            node["cuts"].append((c0, coeffs))
            global_cuts.append((c0, coeffs))
            bulge_added += 1
            added_kind = "bulge"
        else:
            raise ValueError(f"unknown cut_policy: {policy}")

        total_new_cuts += 1
        log_cut(added_kind, log_level)

    # 上限に達した場合
    node["ub"] = float(theta_hat)
    log_warn("このノードでのカット上限に到達。", log_level)
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


# ========== メイン：完全 Branch-and-Cut（論文 Algorithm 1） ====================


def scflp_branch_and_cut(
    data: Dict[str, object],
    max_nodes: int = 30,
    max_rounds_per_node: int = 200,
    tol: float = 1e-12,
    pulp_solver: str = "CBC",
    node_selection: str = "dfs",  # 'dfs' or 'bestbound'
    log_level: str = "info",
    separation: str = "approx",
    cut_policy: str = "auto",
    lp_relax: bool = True,
) -> Dict[str, object]:
    """
    論文 Algorithm 1 に沿った Branch-and-Cut 実装。
    """
    J = int(data["J"])
    p = int(data["p"])
    t0 = time.time()

    # 下界・ベスト解（行2）
    theta_LB = 0.0
    x_best = None

    total_bulge = 0
    total_submod = 0
    total_cuts = 0

    # フォーミュレーション集合 F（行1: 緩和を1つ入れて初期化）
    nodes: List[Dict[str, object]] = [make_node()]
    global_cuts: List[Tuple[float, Dict[int, float]]] = []

    explored = 0
    log_header("Branch-and-Cut（論文 Algorithm 1）開始", log_level)

    while nodes and explored < max_nodes:
        # フォーミュレーション取り出し（行3–4）
        if node_selection == "bestbound":
            idx = int(np.argmax([n.get("ub", -1e100) for n in nodes]))
            node = nodes.pop(idx)
        else:
            node = nodes.pop()

        explored += 1
        log_line(log_level)
        log_step(
            f"🌿 ノード展開: depth={node['depth']}, |fix1|={len(node['fix1'])}, |fix0|={len(node['fix0'])}",
            log_level,
        )

        # 早期チェック: |fix1| > p は不可能
        if len(node["fix1"]) > p:
            log_warn("早期枝刈り: |fix1| > p", log_level)
            continue

        # 行5–9: 違反が無くなるまでこのノードを強化して解く
        info = node_solve_until_no_violation(
            data=data,
            node=node,
            global_cuts=global_cuts,
            pulp_solver=pulp_solver,
            tol=tol,
            cut_policy=cut_policy,
            separation=separation,
            max_cuts_per_node=max_rounds_per_node,
            log_level=log_level,
            lp_relax=True,
        )
        theta_hat = info["theta_hat"]
        x_hat = info["x_hat"]
        is_int = info["integral"]
        node["ub"] = float(theta_hat)

        total_cuts += int(info.get("cuts_added", 0))
        total_bulge += int(info.get("bulge_added", 0))
        total_submod += int(info.get("submod_added", 0))

        # 行10–13: 違反が無くなった後の分岐
        if theta_hat > theta_LB + tol and is_int:
            # 行10–11: インカンベント更新
            theta_LB = float(theta_hat)
            x_best = (x_hat > 0.5).astype(float)
            log_success(f"🎯 インカンベント更新: θ_LB ← {theta_LB:.6f}", log_level)
            continue

        if theta_hat > theta_LB + tol and not is_int:
            # 行12–13: 分枝して 2 つのフォーミュレーションを F へ
            j_star = choose_branch_variable(x_hat)
            if j_star is None:
                # 念のための保険
                theta_LB = max(theta_LB, float(theta_hat))
                x_best = (x_hat > 0.5).astype(float)
                log_warn("x は実質整数だったため更新のみ。", log_level)
                continue

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
            # 早期不可能（|fix1|>p）は投入前に除外
            if len(child1["fix1"]) <= p:
                nodes.append(child1)
            else:
                log_warn(f"子ノード(x[{j_star}]=1)は |fix1|>p のため破棄", log_level)
            nodes.append(child0)
            log_step(
                f"🌱 分枝: j*={j_star} → 子ノード depth={node['depth']+1} を2つ追加",
                log_level,
            )
            continue

        # θ̂ が下界以下なら枝刈り
        log_step(f"🪓 枝刈り: θ̂={theta_hat:.6f} ≤ θ_LB={theta_LB:.6f}", log_level)

    # 終了
    elapsed = time.time() - t0
    gap = 0.0
    if nodes:
        gap = max([n.get("ub", -1e100) for n in nodes]) - theta_LB
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


# =========（参考）以下は高度な厳密分離の実装補助（未使用）======================
# ※ Algorithm 1 に含めないため、使わずに残しています（必要なら呼び出してください）。


def _Ly_S_and_union_allk(
    data: Dict[str, object], S_bin: np.ndarray, y_bin: np.ndarray
) -> Tuple[float, np.ndarray]:
    w = data["w"]
    UL = data["UL"]
    UF = data["UF"]
    h = data["h"]
    S_bin = S_bin.reshape(-1)
    y_bin = y_bin.reshape(-1)
    I, J = w.shape

    wS = w.dot(S_bin)
    wSY = w.dot(np.maximum(S_bin, y_bin))

    num_S = UL + wS
    den_S = UL + UF + wSY
    Ly_S = float((h * (num_S / den_S)).sum())

    add_S = 1.0 - S_bin
    add_SY = 1.0 - np.maximum(S_bin, y_bin)

    num_all = num_S[:, None] + w * add_S[None, :]
    den_all = den_S[:, None] + w * add_SY[None, :]
    Ly_SuK = (h[:, None] * (num_all / den_all)).sum(axis=0)
    return Ly_S, Ly_SuK


def _rho_full_minus_k_vector(
    data: Dict[str, object], y_bin: np.ndarray
) -> Tuple[float, np.ndarray]:
    w = data["w"]
    UL = data["UL"]
    UF = data["UF"]
    h = data["h"]
    y_bin = y_bin.reshape(-1)
    I, J = w.shape

    onesJ = np.ones(J)
    wJ = w.dot(onesJ)
    num_J = UL + wJ
    den_J = UL + UF + wJ
    Ly_J = float((h * (num_J / den_J)).sum())

    y_bar = 1.0 - y_bin
    num_J_minus_k = UL[:, None] + wJ[:, None] - w
    den_J_minus_k = UL[:, None] + UF[:, None] + wJ[:, None] - (y_bar[None, :] * w)
    Ly_J_minus_k = (h[:, None] * (num_J_minus_k / den_J_minus_k)).sum(axis=0)
    rho_full = Ly_J - Ly_J_minus_k
    return Ly_J, rho_full


def _H_value(
    data: Dict[str, object],
    x: np.ndarray,
    y_bin: np.ndarray,
    S_bin: np.ndarray,
    pre: Dict[str, object],
) -> float:
    Ly_S, Ly_SuK = _Ly_S_and_union_allk(data, S_bin, y_bin)
    p = pre["p"]
    rho_full = pre["rho_full"]
    term1 = Ly_S
    term2 = -float(((rho_full * (1.0 - x)) * S_bin).sum())
    term3 = float(x.dot(Ly_SuK)) - p * Ly_S
    return term1 + term2 + term3


def _H_empty(data: Dict[str, object], x: np.ndarray, y_bin: np.ndarray) -> float:
    J = int(data["J"])
    S0 = np.zeros(J)
    Ly_0, Ly_0_u_k = _Ly_S_and_union_allk(data, S0, y_bin)
    p = int(data["p"])
    return float(Ly_0 + x.dot(Ly_0_u_k) - p * Ly_0)


def _greedy_extreme_point_for_baseF(
    data: Dict[str, object],
    x: np.ndarray,
    y_bin: np.ndarray,
    weights: np.ndarray,
    pre: Dict[str, object],
) -> np.ndarray:
    J = int(data["J"])
    order = np.argsort(weights)  # 昇順
    S_bin = np.zeros(J)
    v = np.zeros(J)
    F_prev = 0.0

    for idx in order:
        S_bin[idx] = 1.0
        H_curr = _H_value(data, x, y_bin, S_bin, pre)
        F_curr = H_curr - pre["H_empty"]
        v[idx] = F_curr - F_prev
        F_prev = F_curr

    s = v.sum()
    FJ = _H_value(data, x, y_bin, np.ones(J), pre) - pre["H_empty"]
    if abs(s - FJ) > 1e-9:
        v[order[-1]] += FJ - s
    return v


def _mnp_min_norm_point(
    data: Dict[str, object],
    x: np.ndarray,
    y_bin: np.ndarray,
    pre: Dict[str, object],
    max_iter: int = 60,
    eps: float = 1e-8,
) -> np.ndarray:
    J = int(data["J"])
    rng = np.random.default_rng(0)
    w0 = rng.normal(size=J)
    v0 = _greedy_extreme_point_for_baseF(data, x, y_bin, w0, pre)

    V = [v0]
    lamb = np.array([1.0])
    y = v0.copy()

    for _ in range(max_iter):
        v = _greedy_extreme_point_for_baseF(data, x, y_bin, y, pre)
        if float(y.dot(y - v)) <= eps:
            return y
        V.append(v)
        m = len(V)
        Vmat = np.column_stack(V)
        alpha = np.zeros(m)
        alpha[:-1] = lamb
        alpha[-1] = 0.0

        while True:
            G = Vmat.T @ Vmat
            ones = np.ones(m)
            KKT = np.block([[G, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            rhs = np.zeros(m + 1)
            rhs[-1] = 1.0
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            lam_new = sol[:m]
            if np.all(lam_new >= -1e-12):
                lamb = lam_new
                y = Vmat @ lamb
                break
            dirv = lam_new - alpha
            bad = dirv < 0
            t = np.min(alpha[bad] / (alpha[bad] - lam_new[bad]))
            alpha = alpha + t * (lam_new - alpha)
            keep = alpha > 1e-12
            V = [V[i] for i in range(m) if keep[i]]
            alpha = alpha[keep]
            Vmat = np.column_stack(V)
            m = len(V)
            lamb = alpha / alpha.sum()
            y = Vmat @ lamb

    return y


def submodular_cut_coeffs_fractional_exact(
    data: Dict[str, object],
    x: np.ndarray,
    y_hat: np.ndarray,
) -> Tuple[float, Dict[int, float], float, np.ndarray]:
    """分数 x 用の厳密 NW カット（参考実装; Algorithm 1 では使用しない）。"""
    J = int(data["J"])
    x = x.reshape(J)
    y_bin = (y_hat > 0.5).astype(float).reshape(J)

    Ly_J, rho_full = _rho_full_minus_k_vector(data, y_bin)
    H0 = _H_empty(data, x, y_bin)
    pre = dict(Ly_J=Ly_J, rho_full=rho_full, H_empty=H0, p=int(data["p"]))

    y_mnp = _mnp_min_norm_point(data, x, y_bin, pre)
    S_star = (y_mnp < 0.0 - 1e-12).astype(float)
    S_idx = np.where(S_star > 0.5)[0]

    Ly_S, Ly_SuK = _Ly_S_and_union_allk(data, S_star, y_bin)
    coeffs: Dict[int, float] = {}
    for k in S_idx:
        coeffs[int(k)] = float(rho_full[k])
    notS = np.where(S_star < 0.5)[0]
    rho_S_k = Ly_SuK[notS] - Ly_S
    for k, val in zip(notS, rho_S_k):
        coeffs[int(k)] = float(val)

    c0 = float(Ly_S - rho_full[S_idx].sum())
    rhs_val = c0 + float(sum(coeffs[j] * x[j] for j in range(J)))
    return c0, coeffs, rhs_val, S_idx
