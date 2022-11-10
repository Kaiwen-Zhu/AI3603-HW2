from cliff_walk_sarsa_sens import plan
import numpy as np

for df in np.arange(0.7, 1.01, 0.1):
    for lr in np.arange(0.1, 0.41, 0.1):
        df = round(df, 1)
        lr = round(lr, 1)
        plan(df, lr, 0.99999)