#!/usr/bin/env python
# coding: utf-8

# In[29]:


import scflp_v3 as scflp

import importlib
importlib.reload(scflp)


# In[ ]:


data = scflp.make_random_instance(I=20, J=20, p=2, r=3, beta=0.1, seed=0)


# In[ ]:


result = scflp.scflp_branch_and_cut(
    data,
    max_nodes=20000,
    max_rounds_per_node=100,
    tol=1e-20,
    pulp_solver="CBC",    # GLPK も可
    node_selection="bestbound", # "bestbound" も選べる
    log_level="debug",     # "debug" なら詳細ログ
)

