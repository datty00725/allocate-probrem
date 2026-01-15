
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
