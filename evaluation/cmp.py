import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input', nargs=2)

df1, df2 = parser.parse_args().input

df1 = pd.read_pickle(df1)
df2 = pd.read_pickle(df2)

df1 = df1.sort_values(by=['file_name', 'frame_number'])
#df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
df2 = df2.sort_values(by=['file_name', 'frame_number'])

assert df1.shape == df2.shape

scores = []

for i in range(df1.shape[0]):
    file_name1 = df1['file_name'].iloc[i].split('.')[0]
    file_name2 = df2['file_name'].iloc[i].split('.')[0]
    assert file_name1 == file_name2
    assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
 
    scores.append(np.sum(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float) ** 2))

print ("Average difference: %s" % np.mean(scores))
