
# coding: utf-8

# In[35]:

## Random forest classifying based on two training models
## One uses vectorized 5000 features, one uses Readability features
## vectorized 5000 features has a much better performance with homogeneity around 80%

get_ipython().magic('matplotlib inline')
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np


# #import vectorized feature arrays for acts and scenes
act_Vect = pd.read_csv('VectorizedFeatures/AllComplied/AllAct_Features_Vectorize.txt',sep=',',header = None)
scene_Vect = pd.read_csv('VectorizedFeatures/AllComplied/AllScene_Features_Vectorize.txt',sep=',',header = None)
play_Vect = pd.read_csv('VectorizedFeatures/AllComplied/ALLPlay_Features_Vectorize.txt',sep=',',header = None)

#import LSA reduced vectorized feature arrays for acts and scenes
act_Vect_LSA = pd.read_csv('LSA_VectorizedFeatures/Act_Features_Vectorize_LSA.txt',sep=',',header = None)
scene_Vect_LSA = pd.read_csv('LSA_VectorizedFeatures/Scene_Features_Vectorize_LSA.txt',sep=',',header = None )
play_Vect_LSA = pd.read_csv('LSA_VectorizedFeatures/Play_Features_Vectorize_LSA.txt',sep=',',header = None )

#import readability features
act_Readable = pd.read_csv('Readability_Features/Act_Readability_Feature.txt',sep=',',header = None)
scene_Readable = pd.read_csv('Readability_Features/Scene_Readability_Feature.txt',sep=',',header = None )
play_Readable = pd.read_csv('Readability_Features/Play_Readability_Feature.txt',sep=',',header = None )

#import labels
act_Labels = pd.read_csv('LSA_VectorizedFeatures/act_feature_labels.txt',sep='\n',header = None)
scene_Labels= pd.read_csv('LSA_VectorizedFeatures/scene_feature_labels.txt',sep='\n',header = None )
play_Labels= pd.read_csv('LSA_VectorizedFeatures/play_feature_labels.txt',sep='\n',header = None )

# print(scene_Labels)
act_genre = pd.read_csv('RandomForests/act_genre.txt',sep='\n',header = None)
scene_genre= pd.read_csv('RandomForests/scene_genre.txt',sep='\n',header = None )
play_genre= pd.read_csv('RandomForests/play_genre.txt',sep='\n',header = None )


# In[36]:

## Remove brackets for those genre
act_genre = act_genre[0].map(lambda x: x.strip('[').strip(']'))
play_genre = play_genre[0].map(lambda x: x.strip('[').strip(']'))
scene_genre = scene_genre[0].map(lambda x: x.strip('[').strip(']'))


# In[37]:

from scipy.spatial import distance

## Calculate euclidean distance among  scenes
zero = [0] * 5000
dist = []

## Calculate for plays
for i in range(len(scene_Vect)):
    dst = distance.euclidean(zero,scene_Vect[i:i+1])
    dist.append(dst)
    
dist_scene = {}

string = ""

for i in range(len(dist)):
    dist_scene[i] = dist[i]
print("\n The euclidean distance  of each play are :\n")
print(dist_scene)


# In[39]:

## split the training and testing data sets, as well as genre labels
seq = list(range(0,752))
vect_train, vect_test, readable_train, readable_test, genre_train, genre_test,dist_train,dist_test,seq_train,seq_test = train_test_split(scene_Vect,scene_Readable, scene_genre,dist_scene,seq,test_size=0.5, random_state=42)


# In[9]:

## split the sequence numbers

# seq_train,seq_test = train_test_split(seq,test_size=0.5, random_state=42)


# In[55]:

print ("Training the random forest using scene vectorized data...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 10) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest_vect = forest.fit(vect_train, genre_train)
results_vect = forest.predict(vect_test)
# print(results)
# print("the actual labels are\n")
# print(genre_test)


# In[56]:

from sklearn import metrics
# results = results.map(lambda x: x.strip('[').strip(']'))
predict_results_vect = []
for j in range(len(results_vect)):
    predict_results_vect.insert(-1,float(results_vect[j].strip('[').strip(']')))


## Convert actual results to float array for homogeneity comparison
# r = genre_test[0].map(lambda x: x.strip('[').strip(']'))
r = genre_test
r = np.array(r)
actual_results = []
for i in range(len(r)):
    actual_results.insert(-1,float(r[i]))

print("the predicted results using vectorized features trainig model are\n")
print(predict_results_vect)
print("the actual results are \n")
print(actual_results)
# b = [1,2,4,4]
print("Analysis of prediction results using vectorized features trainig model are\n")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(predict_results_vect, actual_results))
print("Completeness: %0.3f" % metrics.completeness_score(predict_results_vect, actual_results))
print("V-measure: %0.3f" % metrics.v_measure_score(predict_results_vect, actual_results))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(predict_results_vect, actual_results))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(scene_Vect, predict_results, sample_size=1000))


# In[13]:

print ("Training the random forest using scene readability feature data...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest_read = forest.fit(readable_train, genre_train)
results_read = forest.predict(readable_test)


# In[14]:

# results = results.map(lambda x: x.strip('[').strip(']'))
predict_results_read = []
for j in range(len(results_read)):
    predict_results_read.insert(-1,float(results_read[j].strip('[').strip(']')))


## Convert actual results to float array for homogeneity comparison
# r = genre_test[0].map(lambda x: x.strip('[').strip(']'))
# r = np.array(r)
# actual_results = []
# for i in range(len(r)):
#     actual_results.insert(-1,float(r[i]))

print("the predicted results using vectorized features trainig model are\n")
print(predict_results_read)
print("the actual results are \n")
print(actual_results)
print("Analysis of prediction results using vectorized features trainig model are\n")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(predict_results_read, actual_results))
print("Completeness: %0.3f" % metrics.completeness_score(predict_results_read, actual_results))
print("V-measure: %0.3f" % metrics.v_measure_score(predict_results_read, actual_results))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(predict_results_read, actual_results))


# In[43]:

# Creat a distance of the test files dictionary and then sorted them
dist_test_dict = {}

for i in range(len(dist_test)):
    dist_test_dict[i] = dist_test[i]

import operator
sorted_dist_test = sorted(dist_test_dict.items(), key=operator.itemgetter(1))
print(sorted_dist_test)


# In[65]:

# Calcualte the average accuracy based on the number of neighbors
neib = 5
accuracy = []


# print(sorted_dist_test[0])
for i in range(len(sorted_dist_test)): # Loop through all distance test array
    window = [] # A slide window which has the size of neib plus 1
    for j in range(neib + 1):
#         while j+i < len(sorted_dist_test):
        window.append(sorted_dist_test[j+i]) # fill the slide window
    result = [] # array to hold the predicted genre
    actual = [] # array to hold the actual genre
#     print(window)
    for (x, y) in window:
        result.insert(-1,predict_results_vect[x])
        actual.insert(-1,actual_results[x])
#         result.append[predict_results_vect[x]]
#         actual.append[actual_results[x]]
    accuracy.insert(-1,*metrics.homogeneity_score(actual, result))


# In[66]:

# x = window[1]
# print(x)0.949333333333, 0.90222814746,0.845654325925,0.800576662234,0.775901210531,0.747490140506
# 0.725        1
# 0.689026061  3
# 0.645820983  7
# 0.61139545   15
# 0.59255096   31
# 0.570853601  63

# Print the average accuracy
print (np.mean(accuracy))
print(accuracy)

