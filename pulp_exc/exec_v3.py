#!/usr/bin/env python
# coding: utf-8

# In[85]:


import scflp_v4 as scflp

import importlib
importlib.reload(scflp)


# In[86]:


data = scflp.make_random_instance(I=100, J=100, p=2, r=2, beta=0.1, seed=0)


# In[ ]:


res = scflp.scflp_branch_and_cut(
    data,
    max_nodes=1000,
    max_rounds_per_node=3,
    tol=1e-10,
    pulp_solver="CBC",
    node_selection="bestbound",   # "dfs" or "bestbound"
    log_level="info",
    separation="approx",
    cut_policy="auto",
)
print("best theta:", res["theta_best"])
print("best x:", res["x_best"])

