import pandas as pd
import numpy as np

runid = 188

df = pd.read_csv("results_ucb" + str(runid) + ".csv")

print(df.rewards.mean())
