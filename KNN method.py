# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:35:02 2024

@author: User
"""

import pandas as pd
import numpy as np
from collections import Counter as Counter
#load data
spam_test=pd.read_csv(r'C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW2_dataset\spam_test.csv')
spam_train=pd.read_csv(r'C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW2_dataset\spam_train.csv')

#seperate x, y
test_x=spam_test.iloc[:,1:-1].values
test_y=spam_test.iloc[:,-1].values
train_x=spam_train.iloc[:,:-1].values
train_y=spam_train.iloc[:,-1].values

# Check for missing values
print(f'Missing values in test_x: {np.isnan(test_x).any()}')
print(f'Missing values in test_y: {np.isnan(test_y).any()}')


def euclidean_dist(x,y):
    dist=np.sqrt(np.sum((x-y)**2))
    return dist
def knn(train_x,train_y,test_x,k):
    predictions=[]
    for test_point in test_x:
        distances=[]
        for train_point in train_x:
            distance = euclidean_dist(test_point, train_point)
            distances.append(distance)
        nearest_neighbors = np.argsort(distances)[:k]
        
        y_labels=train_y[nearest_neighbors]
        #count the occurances of each y_label
        #The result is a list with one tuple, like [(1,)] if label 1 appears n times among the neighbors
        #[0] -> extract [(1,n)] to (1,)
        #[0] -> extract (1,n) -> 1
        majority_vote=Counter(y_labels).most_common(1)[0][0]
        predictions.append(majority_vote)
    return predictions

def accuracy_score(true_y,pred_y):    
    # Calculate the accuracy
    correct_amt = np.sum(true_y == pred_y)
    return correct_amt / len(true_y)

def Z_score_normalization(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis =0)
    return (data-mean)/std

k_values=[1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
# #report accuracies for different k without normalization
# for k in k_values:
#     pred_y=knn(train_x,train_y,test_x,k)
#     accuracy=accuracy_score(test_y, pred_y)
#     print(f'Without normalizing the data, accuracy with k={k}: {accuracy:.4f} ')
    
#normalization apply
train_x_normalized=Z_score_normalization(train_x)
test_x_normalized=Z_score_normalization(test_x)

# #report accuracies for different k with normalization
# for k in k_values:
#     pred_y_normalized=knn(train_x_normalized,train_y,test_x_normalized,k)
#     accuracy_normalized=accuracy_score(test_y, pred_y_normalized)
#     print(f'With normalizing the data, accuracy with k={k}: {accuracy_normalized:.4f} ')

#output labels for the first 50 instances
print('Predicted labels for the first 50 instances:')
for i in range(50):
    print('\n')
    label_predictions = []
    for k in k_values:
        # Only pass the current test instance
        label = knn(train_x_normalized, train_y, test_x_normalized[i:i+1], k)[0]
        if label == 1:
            label_predictions.append('spam')
        else:
            label_predictions.append('no')
    print(f't{i + 1} {label_predictions}')
