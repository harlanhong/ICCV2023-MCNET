import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input', nargs=1)

df1 = parser.parse_args().input[0]

df1 = pd.read_pickle(df1)
df1 = df1.sort_values(by=['file_name', 'frame_number'])

missing = []

for i in range(df1.shape[0]):
    prediction = df1['value'].iloc[i] 
    if df1['frame_number'].iloc[i] >= 32:
        continue
    missing_prediction = (prediction[:, 0] == -1)
    missing.append(missing_prediction.sum())

print ("Missing count: %s" % np.mean(missing))
