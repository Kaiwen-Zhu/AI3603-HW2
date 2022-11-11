from cliff_walk_qlearning_sens import q_plan
from cliff_walk_sarsa_sens import sarsa_plan
import numpy as np

# for df in np.arange(0.5, 1.01, 0.1):
#     for lr in np.arange(0.1, 0.31, 0.1):
#         df = round(df, 1)
#         lr = round(lr, 1)
#         sarsa_plan(df, lr, 0.99999)
#         # q_plan(df, lr, 0.999)

df = 1
for lr in np.arange(0.1, 0.31, 0.1):
    lr = round(lr, 1)
    sarsa_plan(df, lr, 0.99999)