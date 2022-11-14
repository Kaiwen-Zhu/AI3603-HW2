import pandas as pd

rewards = pd.read_csv("LunarLanderData/LunarLander_v3.csv")
# print(rewards.describe())
last = rewards.tail(10)
print(last.describe())