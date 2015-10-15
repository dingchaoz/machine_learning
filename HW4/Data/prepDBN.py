### convert data to training and testing for Deep Belief Networks
### label to predict genre of each speech
### do 50:50 training:testing

import pandas as pd
import os

## import data
df = pd.read_csv('all_plays.txt', sep='\t')

## create folders from genre
genres = df['genre'].unique()
for g in genres:
    train_path = os.path.join('DBN','train',str(g))
    test_path = os.path.join('DBN','test',str(g))
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

## write to genre
for i in range(len(df)):
    genre = df['genre'].ix[i]
    sp = df['speech'].ix[i]
    if (i + 1) % 2 == 0: #write to train or test
        with open(os.path.join('DBN', 'train', str(genre), str(i + 1) + '.txt'), 'w') as f:
            f.write(sp)
    else:
        with open(os.path.join('DBN', 'test', str(genre), str(i + 1) + '.txt'), 'w') as f:
            f.write(sp)
