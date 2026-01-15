#!/usr/bin/env python
# coding: utf-8

# ## 元論文

# In[9]:


import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os

sys.path.append(os.path.abspath(".."))  # 1階層上を追加
from pulp_exc.scflp_v2 import SCFLPData, SCFLPSolver, make_random_instance


# In[ ]:


"""instances = [
    {"I": 200,  "J": 200,  "p": 2, "r": 2},
    {"I": 300,  "J": 300,  "p": 2, "r": 2},
    {"I": 400,  "J": 400,  "p": 2, "r": 2},
    {"I": 500,  "J": 500,  "p": 2, "r": 2},
    {"I": 600,  "J": 600,  "p": 2, "r": 2},
    {"I": 700,  "J": 700,  "p": 2, "r": 2},
    {"I": 800,  "J": 800,  "p": 2, "r": 2},
    {"I": 900,  "J": 900,  "p": 2, "r": 2},
    {"I": 1000,  "J": 1000,  "p": 2, "r": 2},
]
"""

instances = [
    {"I": 20,  "J": 20,  "p": 2, "r": 2},
    {"I": 20,  "J": 20,  "p": 3, "r": 2},
    {"I": 20,  "J": 20,  "p": 2, "r": 3},

    {"I": 40,  "J": 40,  "p": 2, "r": 2},
    {"I": 40,  "J": 40,  "p": 3, "r": 2},
    {"I": 40,  "J": 40,  "p": 2, "r": 3},

    {"I": 60,  "J": 60,  "p": 2, "r": 2},
    {"I": 60,  "J": 60,  "p": 3, "r": 2},
    {"I": 60,  "J": 60,  "p": 2, "r": 3},

    {"I": 80,  "J": 80,  "p": 2, "r": 2},
    {"I": 80,  "J": 80,  "p": 3, "r": 2},
    {"I": 80,  "J": 80,  "p": 2, "r": 3},

    {"I": 100, "J": 100, "p": 2, "r": 2},
    {"I": 100, "J": 100, "p": 3, "r": 2},
    {"I": 100, "J": 100, "p": 2, "r": 3},
]

"""instances_2 = [
    {"I": 200,  "J": 200,  "p": 2, "r": 2},
    {"I": 300,  "J": 300,  "p": 2, "r": 2},
    {"I": 400,  "J": 400,  "p": 2, "r": 2},
    {"I": 500,  "J": 500,  "p": 2, "r": 2},
]"""


# In[11]:


n_runs = 10
beta = 0.1
all_results = []

for inst_id, params in enumerate(instances, start=1):
    for run in range(n_runs):
        seed = np.random.SeedSequence()
        data = make_random_instance(**params, beta=beta, seed=seed)
        
        solver = SCFLPSolver(data, milp_backend="pulp", pulp_solver="CBC")
        result = solver.solve(max_rounds=2000, verbose=False)

        # 各試行ごとに記録
        row = {
            "instance_id": inst_id,
            "run": run,
            "I": params["I"],
            "J": params["J"],
            "p": params["p"],
            "r": params["r"],
            "beta": beta,
            "theta": result["theta"],
            "time_sec": result["time_sec"],
            "status": result.get("status", None),
            "iterations": result.get("iterations", None),
            "cuts": result.get("cuts", None),
            "obj": result.get("obj", None),
        }
        all_results.append(row)

# DataFrame にまとめて CSV 保存
df = pd.DataFrame(all_results)
df.to_csv("scflp_results_all_over100_with_previous_to1000.csv", index=False, encoding="utf-8")

print("保存完了: scflp_results_all_over100_with_previous_to1000.csv")


# ## 豊島

# In[12]:


get_ipython().run_line_magic('run', '..\\mutaion_gda\\func_v4.ipynb')


# In[13]:


import numpy as np
import pandas as pd
import random, time

results = []
num_rows_columns = 50   # サイト配置グリッドの一辺
alpha = 0.0             # 固定
beta = 0.1
n_runs = 10             # 繰り返し回数

for inst_id, params in enumerate(instances, start=1):
    objs_relaxed, objs_binary, objs_ex, times = [], [], [], []

    for run in range(n_runs):
        seed_int = random.getrandbits(32)

        # 需要点の重みベクトル
        h_i = np.full(params["I"], 1 / params["I"])
        J_L, J_F = set(), set()

        start = time.time()
        x_R, y_R, obj_relaxed, x_proj, y_proj, obj_binary, obj_ex, candidate_sites, demand_points, history = lgda_solver(
            params["I"], params["J"], num_rows_columns,
            params["p"], params["r"],
            alpha, beta, h_i, J_L, J_F,
            eta_x=0.01, eta_y=0.01,
            mu=.5,
            max_iter=100_000,
            tau_interval=2000,
            return_history=True
        )
        elapsed = time.time() - start

        objs_relaxed.append(obj_relaxed)
        objs_binary.append(obj_binary)
        objs_ex.append(obj_ex)
        times.append(elapsed)

    # 各インスタンスの平均を保存
    row = {
        "I": params["I"],
        "J": params["J"],
        "p": params["p"],
        "r": params["r"],
        #"beta": beta,
        "obj_relaxed_avg": np.mean(objs_relaxed),
        "obj_binary_avg": np.mean(objs_binary),
        "obj_ex_avg": np.mean(objs_ex),
        "time_avg_sec": np.mean(times)
    }
    results.append(row)

# DataFrame 化 & CSV 保存
df = pd.DataFrame(results)
df.to_csv("lgda_results_avg_over100_with_previous_to1000.csv", index=False, encoding="utf-8")
print(df)
print("保存完了: lgda_results_avg_over100_with_previous_to1000.csv")


# In[15]:


import pandas as pd
from pathlib import Path

# 入出力パスを指定
in_path = Path("result.csv")              # ← 元のCSVパスに置き換え
out_path = Path("results_rounded.csv")     # ← 出力CSV名

# 四捨五入する列
target_cols = ["obj(MIP)", "obj(Mutation)"]
sec = ["time(MIP)", "time(Mutation)"]
# 読み込み
df = pd.read_csv(in_path)

# 必要列の存在チェック
missing = [c for c in target_cols if c not in df.columns]
if missing:
    raise ValueError(f"次の列が見つかりません: {missing}")

# 数値化して小数第4位で四捨五入
for c in target_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    
# 数値化して小数第4位で四捨五入
for c in sec:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)

# 保存（インデックス列は付けない）
df.to_csv(out_path, index=False)

print(f"保存しました: {out_path.resolve()}")

