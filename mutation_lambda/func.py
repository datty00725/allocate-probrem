#!/usr/bin/env python
# coding: utf-8

# # L0

# In[ ]:


def lgda_solver(
    D: int, J: int, num_rows_columns: int,
    p: int, r: int,
    alpha: float, beta: float,
    h: np.ndarray, J_L: set, J_F: set,
    *,  # ここから下はキーワード専用
    lambda0: np.ndarray = None,
    eta_x: float = 0.01, eta_y: float = 0.01, eta_lambda: float = 0.01,
    mu: float = 0.01,
    max_iter: int = 500, tau_interval: int = 10,  # ← N = tau_interval
    tol: float = 1e-6,
    enforce_no_overlap: bool = False,
    return_history: bool = False,
    fix_seed: bool = False,
    seed: int = 42
):
    """LGDA with Mutation. 各反復では連続緩和に射影、終了時に 0/1 射影。
    基準戦略 cp,cq は t が N,2N,3N,... のときに p(t),q(t) へ更新。
    """
    if fix_seed is True:
        seed_int = seed
    else:
        rng = np.random.default_rng()
        seed_int = int(rng.integers(0, 2**32, dtype=np.uint64))  # 上端は排他的
    print(seed_int)
    # インスタンス生成
    demand_points, candidate_sites = generate_instance(num_rows_columns, D, J, seed=seed_int)
    distances = compute_distances(demand_points, candidate_sites)
    w = compute_wij_matrix(distances, alpha, beta)
    Ui_L = compute_Ui_L(w, J_L)
    Ui_F = compute_Ui_F(w, J_F)
    
    x0 = np.random.rand(len(candidate_sites))  # 例: 初期値
    y0 = np.random.rand(len(candidate_sites))  # 同じ次元

    # 初期化（可行域へ）
    x = project_box_l1(np.clip(x0, 0, 1), p)
    #y = project_box_l1(np.clip(y0, 0, 1), r, mask=(x > 0) if enforce_no_overlap else None)
    y = project_box_l1(np.clip(y0, 0, 1), r)

    if lambda0 is None:
        lam = np.zeros(J)
    else:
        lam = np.maximum(lambda0.copy(), 0.0)

    # 基準戦略（時刻0の戦略で初期化）
    cp, cq = x.copy(), y.copy()

    obj_vals, dx_vals, dy_vals = [], [], []
    prev_obj = None

    # 反復：t = 1..max_iter
    for t in range(1, max_iter + 1):
        # gx = grad_x(x, y, w, Ui_L, Ui_F, h)  # ∇_x L（上昇方向に用いる）
        # gy = grad_y(x, y, w, Ui_L, Ui_F, h)  # ∇_y L（下降方向に用いる）
        gx = grad_x_tilde(x, y, w, Ui_L, Ui_F, h)   # ∇_x \tilde L
        gy = grad_y_tilde(x, y, w, Ui_L, Ui_F, h)   # ∇_y \tilde L

        # Mutation 付き更新：x 上昇, y 下降 + 参照点への引き戻し
        # x_tmp = x + eta_x * (gx - mu * (x -cp))
        # y_tmp = y + eta_y * (-gy - mu * (y -cq))
        x_tmp = x + eta_x * (gx + lam - mu * (x - cp))
        y_tmp = y + eta_y * (-gy + lam - mu * (y - cq))


        # 連続緩和の集合へ射影
        x_next = project_box_l1(x_tmp, p)
        y_next = project_box_l1(y_tmp, r)
        lam = np.maximum(lam + eta_lambda * (x_next + y_next - 1.0), 0.0)

        dx = np.linalg.norm(x_next - x)
        dy = np.linalg.norm(y_next - y)

        # new_obj = compute_Lhat(x_next, y_next, w, Ui_L, Ui_F, h)
        new_obj = compute_Ltilde(x_next, y_next, w, Ui_L, Ui_F, h)
        obj_vals.append(new_obj)
        dx_vals.append(dx)
        dy_vals.append(dy)

        # 状態更新
        x, y = x_next, y_next

        # 収束判定（必要なら有効化）
        """
        if (max(dx, dy) < tol) or (prev_obj is not None and abs(new_obj - prev_obj) < tol):
            prev_obj = new_obj
            # t が N の倍数なら、仕様通り cp,cq を p(t),q(t) に更新してから抜ける
            if (t % tau_interval) == 0:
                cp, cq = x.copy(), y.copy()
            break
        """
        prev_obj = new_obj

        # 時刻 N ごとに cp,cq を今日の戦略 p(t), q(t) に更新
        if (t % tau_interval) == 0:
            cp, cq = x.copy(), y.copy()
            # print("copy")

    # 終了時に 0/1 射影で確定解
    x_proj = project_cardinality(x, p)
    # y_proj = project_cardinality(y, r, mask=(x_proj > 0) if enforce_no_overlap else None)
    y_proj = project_cardinality(y, r)

    obj_final_relaxed = compute_Lhat(x, y, w, Ui_L, Ui_F, h)          # 連続版
    obj_final_binary  = compute_Lhat(x_proj, y_proj, w, Ui_L, Ui_F, h) # 0/1版
    obj_final_ex      = compute_L(h, Ui_L, Ui_F, w, x_proj, y_proj)    # 別定義がある場合

    history = {"objective": np.array(obj_vals), "dx": np.array(dx_vals), "dy": np.array(dy_vals)}

    if return_history:
        return x, y, obj_final_relaxed, x_proj, y_proj, obj_final_binary, obj_final_ex, candidate_sites, demand_points, history
    return x, y, obj_final_relaxed, x_proj, y_proj, obj_final_binary, obj_final_ex, candidate_sites, demand_points


# # L1

# In[ ]:


def generate_instance(num_rows_columns, I, J, seed=42):
    """
    Generate I demand points and J candidate facility sites from a grid of size num_rows_columns x num_rows_columns.
    Returns: (demand_points, candidate_sites)
    """
    if seed is not None:
        random.seed(seed)
    
    # すべての格子点を生成（1始まり）
    all_points = [(x, y) for x in range(1, num_rows_columns + 1)
                         for y in range(1, num_rows_columns + 1)]
    
    # ランダムにシャッフル
    random.shuffle(all_points)
    
    # 十分な点があるか確認
    assert I + J <= len(all_points), "Grid is too small for given I and J."
    
    demand_points = all_points[:I]
    candidate_sites = all_points[I:I+J]
    
    return demand_points, candidate_sites


# In[ ]:


def compute_Ui_L(wij_matrix, J_L):
    """
    リーダーの既存施設による U_i^L を計算する関数

    Parameters:
        wij_matrix (np.array): D × J の w_ij の重み行列
        J_L (set): リーダーが既に持っている施設のインデックス集合

    Returns:
        np.array: 各需要点 i に対する U_i^L のベクトル
    """
    D, _ = wij_matrix.shape  # D: 需要点の数, J: 候補施設の数

    # J_L が空なら影響はゼロ
    if not J_L:
        return np.zeros(D)

    # 各需要点 i に対して、リーダーの施設 j ∈ J_L からの重みを合計する
    utility_vector = np.zeros(D)
    for j in J_L:
        # 列 j は、施設 j が各需要点に与える重み
        utility_vector += wij_matrix[:, j]

    return utility_vector


# In[ ]:


def compute_Ui_F(wij_matrix, J_F):
    """
    リーダーの既存施設による U_i^L を計算する関数

    Parameters:
        wij_matrix (np.array): D × J の w_ij の重み行列
        J_L (set): リーダーが既に持っている施設のインデックス集合

    Returns:
        np.array: 各需要点 i に対する U_i^L のベクトル
    """
    D, _ = wij_matrix.shape  # D: 需要点の数, J: 候補施設の数

    # J_L が空なら影響はゼロ
    if not J_F:
        return np.zeros(D)

    # 各需要点 i に対して、リーダーの施設 j ∈ J_L からの重みを合計する
    utility_vector = np.zeros(D)
    for j in J_F:
        # 列 j は、施設 j が各需要点に与える重み
        utility_vector += wij_matrix[:, j]

    return utility_vector


# In[ ]:


def compute_wij_matrix(distances, alpha=0, beta=0.1):
    wij_matrix = np.exp(alpha - beta * distances)
    return wij_matrix


# In[ ]:


def compute_distances(demand_points, candidate_sites):
    D, J = len(demand_points), len(candidate_sites)  # ここで D, J を定義
    distances = np.zeros((D, J))
    for d in range(D):
        for j in range(J):
            distances[d, j] = np.sqrt(
                (demand_points[d][0] - candidate_sites[j][0]) ** 2
                + (demand_points[d][1] - candidate_sites[j][1]) ** 2
            )
    return distances


# In[ ]:


def grad_x(x: np.ndarray, y: np.ndarray, w: np.ndarray,
           Ui_L: np.ndarray, Ui_F: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Gradient of \hat{L} with respect to x (ascent direction)."""
    Ai = compute_Ai(x, y, w, Ui_L)
    Bi = compute_Bi(x, y, w, Ui_L, Ui_F)

    dA_dx = w * ((1.0 + y) - 2.0 * y * x)     # shape (I, J)
    dB_dx = w * (1.0 - y)

    frac = (Bi[:, None] * dA_dx - Ai[:, None] * dB_dx) / (Bi[:, None] ** 2) + 1e-8
    return (frac * h[:, None]).sum(axis=0)


# In[ ]:


def grad_y(x: np.ndarray, y: np.ndarray, w: np.ndarray,
           Ui_L: np.ndarray, Ui_F: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Gradient of \hat{L} with respect to y (descent direction)."""
    Ai = compute_Ai(x, y, w, Ui_L)
    Bi = compute_Bi(x, y, w, Ui_L, Ui_F)

    dA_dy = w * (-x ** 2 + x)
    dB_dy = w * (1.0 - x)

    frac = (Bi[:, None] * dA_dy - Ai[:, None] * dB_dy) / (Bi[:, None] ** 2) + 1e-8
    return (frac * h[:, None]).sum(axis=0)


# In[ ]:


def compute_Lhat(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                 Ui_L: np.ndarray, Ui_F: np.ndarray, h: np.ndarray) -> float:
    """Evaluate the objective \hat{L}(x, y)."""
    Ai = compute_Ai(x, y, w, Ui_L)
    Bi = compute_Bi(x, y, w, Ui_L, Ui_F)
    return float(np.dot(h, Ai / Bi))


# In[ ]:


def project_cardinality(v: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Project a vector onto the set {0,1}^J with at most *k* ones."""
    v = np.clip(v, 0.0, 1.0)
    if mask is not None:
        v = v * (~mask)

    if k >= v.size:
        return (v > 0).astype(float)

    idx = np.argpartition(-v, k)[:k]
    out = np.zeros_like(v)
    out[idx] = 1.0
    return out


# In[ ]:


def project_box_l1(v: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Project v onto { z in [0,1]^n : sum(z) <= k }.
    If mask is given (bool array), force z[mask]=0 before projection.

    project_cardinality関数との違いだが、こちらは連続値で [0,1] の範囲にクリップしてから、合計が k 以下になるように調整する。
    """
    z = v.copy()

    # 0 固定マスク
    if mask is not None:
        z = z.copy()
        z[mask] = 0.0

    # まず [0,1] にクリップ
    z = np.clip(z, 0.0, 1.0)

    s = z.sum()
    if s <= k:
        return z  # 既に可行

    # 合計が大きい場合：tau を探して z = clip(v - tau, 0, 1) の合計を k にする
    # 単峰・区分線形なので 2 分探索で十分安定
    lo, hi = -1.0, 1.0
    # 範囲を拡張してカバー
    vmax = np.max(v)
    vmin = np.min(v)
    lo = vmin - 1.0
    hi = vmax

    def S(tau):
        return np.clip(v - tau, 0.0, 1.0).sum()

    for _ in range(60):  # 1e-18 くらいまで収束
        mid = (lo + hi) / 2.0
        if S(mid) > k:
            lo = mid
        else:
            hi = mid
    z = np.clip(v - hi, 0.0, 1.0)

    if mask is not None:
        z[mask] = 0.0
    return z


# ### Ltilde用の関数

# In[1]:


def compute_Ltilde(x, y, w, Ui_L, Ui_F, h):
    A = Ui_L + (w @ x)
    B = Ui_L + Ui_F + (w @ (x + y))
    B = np.maximum(B, 1e-12)
    return float(np.dot(h, A / B))

def grad_x_tilde(x, y, w, Ui_L, Ui_F, h):
    A = Ui_L + (w @ x)
    B = Ui_L + Ui_F + (w @ (x + y))
    B = np.maximum(B, 1e-12)
    tmp = h * (Ui_F + (w @ y)) / (B**2)
    return w.T @ tmp

def grad_y_tilde(x, y, w, Ui_L, Ui_F, h):
    A = Ui_L + (w @ x)
    B = Ui_L + Ui_F + (w @ (x + y))
    B = np.maximum(B, 1e-12)
    tmp = h * A / (B**2)
    return -(w.T @ tmp)


# # L2

# In[ ]:


def compute_Ai(x: np.ndarray, y: np.ndarray, w: np.ndarray, Ui_L: np.ndarray) -> np.ndarray:
    """Compute A_i = U_i^L + Σ_j w_ij[-y_j x_j^2 + (1+y_j)x_j] for all i."""
    term = w * ((1.0 + y) * x - y * x ** 2)  # broadcast over j
    Ai = Ui_L + term.sum(axis=1)
    # print("Ai min:", np.min(Ai))
    return Ai


# In[ ]:


def compute_Bi(x: np.ndarray, y: np.ndarray, w: np.ndarray,
               Ui_L: np.ndarray, Ui_F: np.ndarray) -> np.ndarray:
    """Compute B_i = U_i^L + U_i^F + Σ_j w_ij[(1-y_j)x_j + y_j] for all i."""
    term = w * ((1.0 - y) * x + y)
    Bi = Ui_L + Ui_F + term.sum(axis=1)
    # print("Bi min:", np.min(Bi))
    return Bi


# # Prot

# In[ ]:


def plot_minmax_history(L_vals: list | np.ndarray,
                        dx_vals: list | np.ndarray,
                        dy_vals: list | np.ndarray,
                        *, logy: bool = True):
    """Plot convergence history returned by *minmax_solver*.

    Parameters
    ----------
    L_vals : sequence of float
        Objective values (hist_Lcont).
    dx_vals : sequence of float
        Norm of x updates.
    dy_vals : sequence of float
        Norm of y updates.
    logy : bool, default True
        Use log scale for dx/dy curves.
    """
    L_vals = np.asarray(L_vals)
    dx_vals = np.asarray(dx_vals)
    dy_vals = np.asarray(dy_vals)
    iters = np.arange(1, len(L_vals) + 1)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(iters, L_vals, label="objective", linewidth=1.5, color="tab:blue")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("objective")

    ax2 = ax1.twinx()
    ax2.plot(iters, dx_vals, linestyle="--", label="‖Δx‖", color="tab:orange")
    ax2.plot(iters, dy_vals, linestyle=":", label="‖Δy‖", color="tab:green")
    ax2.set_ylabel("step size")
    if logy:
        ax2.set_yscale("log")

    lines, labels = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lab2, loc="best")
    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_facility_selection(candidate_sites, demand_points, x_bin, y_bin):
    """
    施設配置の可視化関数。
    
    Parameters
    ----------
    candidate_sites : list of tuple(float, float)
        候補施設の座標 [(x1, y1), (x2, y2), ...]
    demand_points : list of tuple(float, float)
        需要点の座標 [(x1, y1), (x2, y2), ...]
    x_bin : array-like of 0/1
        リーダーによって選ばれた施設（青丸で表示）
    y_bin : array-like of 0/1
        フォロワーによって選ばれた施設（赤丸で表示）
    """
    # 座標分解
    candidate_x = [pt[0] for pt in candidate_sites]
    candidate_y = [pt[1] for pt in candidate_sites]
    demand_x = [pt[0] for pt in demand_points]
    demand_y = [pt[1] for pt in demand_points]

    # プロット開始
    plt.figure(figsize=(8, 8))

    # 需要点（黒）
    plt.scatter(demand_x, demand_y, color='black', marker='x', label='Demand Points')

    # 候補地（グレー）
    plt.scatter(candidate_x, candidate_y, color='gray', label='Candidate Sites')

    # x_bin == 1 → 青い○（枠のみ）
    for i, val in enumerate(x_bin):
        if val == 1:
            plt.scatter(candidate_sites[i][0], candidate_sites[i][1],
                        s=200, facecolors='none', edgecolors='blue', linewidths=2,
                        label='x_bin = 1' if 'x_bin = 1' not in plt.gca().get_legend_handles_labels()[1] else "")

    # y_bin == 1 → 赤い●
    for i, val in enumerate(y_bin):
        if val == 1:
            plt.scatter(candidate_sites[i][0], candidate_sites[i][1],
                        s=100, color='red',
                        label='y_bin = 1' if 'y_bin = 1' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 軸・凡例など
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Demand Points and Candidate Sites with Selections')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


# In[ ]:


def plot_each_history_component_separately(
    history,
    *,
    logy: bool = True,
    fix_seed: bool = False,
    save: bool = False,
    save_prefix: str = "history",
    save_ext: str = "png",
    dpi: int = 300,
):
    """
    Plot objective, ‖dx‖, and ‖dy‖ from the LGDA solver history as separate figures.

    Parameters
    ----------
    history : dict with keys "objective", "dx", "dy"
        Output from lgda_solver with return_history=True.
    logy : bool
        Whether to use logarithmic scale for ‖dx‖ and ‖dy‖ plots.
    """
    iters = np.arange(1, len(history["objective"]) + 1)

    # 1. Objective
    plt.figure(figsize=(6, 4))
    plt.plot(iters, history["objective"], label="Objective", color="tab:blue", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    # plt.title("Objective over Iterations")
    plt.grid(True)
    # plt.legend()
    if fix_seed is True:
        plt.ylim(0.5,0.545)
        # pass

    if save:
        fname = f"{save_prefix}_objective.{save_ext}"
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        print("save pic:", fname)

    plt.tight_layout()
    plt.show()

    # 2. dx
    plt.figure(figsize=(6, 4))
    plt.plot(iters, history["dx"], label="‖Δx‖", color="tab:orange", linestyle="--", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("‖Δx‖")
    # plt.title("‖dx‖ over Iterations")        
    # plt.ylim(bottom=-1) 
    plt.grid(True)
    # plt.legend()

    if save:
        fname = f"{save_prefix}_dx.{save_ext}"
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        print("save pic:", fname)

    plt.tight_layout()
    plt.show()

    # 3. dy
    plt.figure(figsize=(6, 4))
    plt.plot(iters, history["dy"], label="‖Δy‖", color="tab:green", linestyle=":", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("‖Δy‖")
    # plt.title("‖dy‖ over Iterations")
    #plt.ylim(bottom=-1) 
    plt.grid(True)
    # plt.legend()

    if save:
        fname = f"{save_prefix}_dy.{save_ext}"
        plt.savefig(fname, dpi=dpi, bbox_inches="tight")
        print("save pic:", fname)

    plt.tight_layout()
    plt.show()


# # Optional

# In[ ]:


def compute_L(h_i, Ui_L, Ui_F, wij, x, y):
    """
    関数 L(x, y) を計算する

    Parameters:
        h (np.array): 需要点ごとの人口密度ベクトル (D,)
        Ui_L (np.array): 各需要点におけるリーダーの影響度 (D,)
        Ui_F (np.array): 各需要点におけるフォロワーの影響度 (D,)
        wij (np.array): 需要点と施設候補の重み行列 (D, J)
        x (np.array): リーダーが選択した施設配置 (J,)
        y (np.array): フォロワーが選択した施設配置 (J,)

    Returns:
        float: L(x, y) の計算結果
    """
    numerator = Ui_L + (wij @ x)  # 分子: リーダーの影響度 + 選択した施設の影響
    denominator = Ui_L + Ui_F + (wij @ np.maximum(x, y))  # 分母: 総合影響度

    return np.sum(h_i * (numerator / denominator))


# In[ ]:


def _moving_avg(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([])
    c = np.convolve(x, np.ones(w)/w, mode='valid')
    return c

def history_to_frame(history):
    # Convert history dict {'objective','dx','dy', ...} into a tidy DataFrame with t index.
    keys = sorted(history.keys())
    L = max(len(history[k]) for k in keys)
    data = {}
    for k in keys:
        v = np.asarray(history[k])
        if v.ndim == 0:
            v = np.array([v])
        if len(v) < L:
            pad = np.full(L - len(v), np.nan)
            v = np.concatenate([v, pad])
        data[k] = v
    df = pd.DataFrame(data)
    df.index.name = "t"
    return df

def summarize_history(history, tol=1e-6, window=50):
    # Return a dictionary of simple diagnostics from the history.
    df = history_to_frame(history)
    out = {}
    if "objective" in df:
        obj = df["objective"].values
        out["iters_recorded"] = int(np.sum(~np.isnan(obj)))
        if out["iters_recorded"] > 1:
            rel_changes = np.abs(np.diff(obj)) / (np.maximum(1.0, np.abs(obj[:-1])))
            out["median_rel_obj_change"] = float(np.nanmedian(rel_changes))
            out["last_rel_obj_change"] = float(rel_changes[-1])
            deltas = np.diff(obj)
            sign_flips = np.sum(np.sign(deltas[1:]) * np.sign(deltas[:-1]) < 0)
            out["obj_sign_flips"] = int(sign_flips)
    for k in ("dx", "dy"):
        if k in df:
            s = df[k].values
            out[f"{k}_last"] = float(s[~np.isnan(s)][-1]) if np.any(~np.isnan(s)) else float("nan")
            ma = _moving_avg(np.nan_to_num(s, nan=np.nan), min(window, max(1, np.sum(~np.isnan(s))//2 or 1)))
            out[f"{k}_moving_avg_last"] = float(ma[-1]) if len(ma) else float("nan")
            out[f"{k}_below_tol_ratio"] = float(np.mean(np.nan_to_num(s, nan=np.inf) < tol)) if np.any(~np.isnan(s)) else float("nan")
    out["likely_stagnated"] = bool(
        out.get("dx_below_tol_ratio", 0) > 0.8 and out.get("dy_below_tol_ratio", 0) > 0.8
    )
    out["likely_oscillating"] = bool(out.get("obj_sign_flips", 0) > (out.get("iters_recorded", 0) * 0.2))
    return out

def plot_history(history, tau_interval=None):
    # Make simple plots for objective and step sizes. One plot per figure.
    df = history_to_frame(history)
    if "objective" in df:
        plt.figure()
        df["objective"].plot()
        plt.title("Objective per iteration")
        plt.xlabel("iteration")
        plt.ylabel("objective")
        if tau_interval:
            for t in range(tau_interval, len(df), tau_interval):
                plt.axvline(t, linestyle="--", alpha=0.2)
        plt.show()
    for k in ("dx", "dy"):
        if k in df:
            plt.figure()
            df[k].plot()
            plt.title(f"{k} per iteration")
            plt.xlabel("iteration")
            plt.ylabel(k)
            if tau_interval:
                for t in range(tau_interval, len(df), tau_interval):
                    plt.axvline(t, linestyle="--", alpha=0.2)
            plt.show()

def save_history_csv(history, path="lgda_history.csv"):
    df = history_to_frame(history)
    df.to_csv(path, index=True)
    return path

def mark_tau_events(length, tau_interval):
    return [t for t in range(length) if (t % tau_interval) == 0 and t > 0]


# In[ ]:


import numpy as np

def check_convergence(history, tol=1e-6, patience=50):
    """
    historyから収束判定を行い、
    - 収束していれば 0
    - 収束していなければ 1
    を返す関数。

    Parameters
    ----------
    history : dict
        lgda_solver(..., return_history=True) で得られる辞書
        {"objective": ..., "dx": ..., "dy": ...}
    tol : float
        許容誤差（dx, dy の閾値）
    patience : int
        連続して何回 tol を満たしたら「収束」とみなすか

    Returns
    -------
    int : 0 (収束), 1 (非収束)
    """
    dx = np.asarray(history.get("dx", []))
    dy = np.asarray(history.get("dy", []))

    if len(dx) == 0 or len(dy) == 0:
        return 1  # データがなければ非収束扱い

    # 末尾から patience 回分を確認
    recent_dx = dx[-patience:]
    recent_dy = dy[-patience:]

    # すべて tol 以下なら収束と判定
    if np.all(recent_dx < tol) and np.all(recent_dy < tol):
        return 0
    else:
        return 1

