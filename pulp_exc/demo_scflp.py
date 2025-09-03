# demo_scflp.py
# Example usage of scflp.SCFLPSolver

import numpy as np
from scflp_v2 import SCFLPData, SCFLPSolver, make_random_instance

seed = np.random.SeedSequence()
# 1) Build a toy instance (no pre-existing facilities)
data = make_random_instance(I=20, J=20, p=6, r=2, beta=0.1, seed=seed)

# 2) Solve (PuLP/CBC by default). If you have Gurobi, use milp_backend='gurobi'
solver = SCFLPSolver(data, milp_backend="pulp", pulp_solver="CBC")
result = solver.solve(max_rounds=2000, verbose=True)

print("\n=== RESULT ===")
x = result["x"]
print("Objective (leader market share):", result["obj"])
print("Selected sites (indices where x=1):", np.where(x > 0.5)[0].tolist())
print(
    "theta:",
    result["theta"],
    "sum x:",
    x.sum(),
    "iterations:",
    result["iterations"],
    "cuts:",
    result["cuts"],
)
