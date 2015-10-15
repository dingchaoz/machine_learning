### convert data to training and testing for Deep Belief Networks
### label to predict genre of each work
### do 50:50 training:testing

import pandas as pd
import os

## import data
df = pd.read_csv('all_plays.txt', sep='\t')

## create folders from genre
genres = df['genre'].unique()
for g in genres:
    train_path = os.path.join('DBN2','train',str(g))
    test_path = os.path.join('DBN2','test',str(g))
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)


# Get all the work names
df['name'] = df['playname'] + '_' + df['scene'].map(str) + '_' + df['act'].map(str) + '_' + df['title']
names = pd.unique(df.name.ravel())

# Array to hold bag of words for each work
work_bagwords = []

# Array to hold genre
work_genres = []

# Array to hold labels
work_label = []

## loop through all works
for i in range(len(names)):

    ### Create bag of words for all works
    p = df[df['name'] == names[i]] # Get the sub data frame of each work
    s = "" # Initiate empty string to hold bag of words for work

    # Iterate all the rows to append the speech and speaker words to a string
    for index,row in p.iterrows():
        s += str(row['speech'])
        genre = row['genre'] #get genre

    # Append the bag of words to each work
    work_bagwords.append(s)

    # Append genre for each work
    work_genres.append(genre)

    # Append the label to each row
    work_label.append(str(names[i]))

## write to txt with genre as folder name
for i in range(len(work_bagwords)):
    genre = work_genres[i]
    sp = work_bagwords[i]
    if (i + 1) % 2 == 0: #write to train or test
        with open(os.path.join('DBN2', 'train', str(genre), str(i + 1) + '.txt'), 'w') as f:
            f.write(sp)
    else:
        with open(os.path.join('DBN2', 'test', str(genre), str(i + 1) + '.txt'), 'w') as f:
            f.write(sp)

