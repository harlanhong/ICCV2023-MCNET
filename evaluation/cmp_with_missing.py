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
missing = []

for i in range(df1.shape[0]):
    file_name1 = df1['file_name'].iloc[i].split('.')[0]
    file_name2 = df2['file_name'].iloc[i].split('.')[0]
    assert file_name1 == file_name2
    assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
    gt = df1['value'].iloc[i]
    missing_gt = (gt[:, 0] == -1)
    prediction = df2['value'].iloc[i] 
    missing_prediction = (prediction[:, 0] == -1)

    present = np.logical_and(np.logical_not(missing_gt), np.logical_not(missing_prediction))
    prediction_fail = np.logical_and(np.logical_not(missing_gt), missing_prediction)

#    print (18 - missing_gt.sum())

    present_gt = 18 - missing_gt.sum()
    if present_gt.sum() != 0:
        missing.append(prediction_fail.sum() / present_gt)
        
    if present.sum() != 0:
        scores.append(np.mean(np.abs(gt[present] - prediction[present]).astype(float)))

print ("Average difference: %s" % np.mean(scores))
print ("Missing count: %s" % np.mean(missing))
