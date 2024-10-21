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

#func returns all_distances, list of lists (2D list)
#each inner list correspondes to a test point
# each inner list contains the dist from that test point to all training points
#[
#    [distance_from_test_point_0_to_train_point_0, distance_from_test_point_0_to_train_point_1, distance_from_test_point_0_to_train_point_2],
#    [distance_from_test_point_1_to_train_point_0, distance_from_test_point_1_to_train_point_1, distance_from_test_point_1_to_train_point_2]
#]

def precompute_distances(train_x, test_x):
    all_distances=[]
    for test_point in test_x:
        list_distances=[]
        for train_point in train_x:
            distances = euclidean_dist(test_point, train_point)
            list_distances.append(distances)
        all_distances.append(list_distances)
    return all_distances

precomputed_distances = precompute_distances(train_x, test_x)

def knn(precomputed_distances,train_y,k):
    predictions=[]
    for distances in precomputed_distances:
        nearest_neighbors = np.argsort(distances)[:k]
        
        y_labels=train_y[nearest_neighbors]
        #count the occurances of each y_label
        #The result is a list with one tuple, like [(1,)] if label 1 appears n times among the neighbors
        #[0] -> extract [(1,n)] to (1,)
        #[0] -> extract (1,n) -> 1
        majority_vote=Counter(y_labels).most_common(1)[0][0]
        predictions.append(majority_vote)
    return predictions #list of [1,0,1,0,...]

def accuracy_score(true_y,pred_y):    
    # Calculate the accuracy
    correct_amt = np.sum(true_y == pred_y)
    return correct_amt / len(true_y)

def Z_score_normalization(data):
    mean=np.mean(data,axis=0)
    std=np.std(data,axis =0)
    return (data-mean)/std

k_values=[1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
#report accuracies for different k without normalization
for k in k_values:
    pred_y=knn(precomputed_distances,train_y,k)
    accuracy=accuracy_score(test_y, pred_y)
    print(f'Without normalizing the data, accuracy with k={k}: {accuracy:.4f} ')
    
#normalization apply
train_x_normalized=Z_score_normalization(train_x)
test_x_normalized=Z_score_normalization(test_x)

# Precompute distances for normalized data
precomputed_distances_normalized = precompute_distances(train_x_normalized, test_x_normalized)

# Report accuracies for different k values with normalized data
for k in k_values:
    pred_y_normalized = knn(precomputed_distances_normalized, train_y, k)
    accuracy_normalized = accuracy_score(test_y, pred_y_normalized)
    print(f'With normalizing the data, accuracy with k={k}: {accuracy_normalized:.4f}')

#output labels for the first 50 instances
print('Predicted labels for the first 50 instances:')
for i in range(50):
    label_predictions = []
    for k in k_values:
        # Only pass the current precomputed distances
        distances=precomputed_distances_normalized[i]
        nearest_neighbors = np.argsort(distances)[:k]
        y_labels = train_y[nearest_neighbors]
        label = Counter(y_labels).most_common(1)[0][0]
        if label == 1:
            label_predictions.append('spam')
        else:
            label_predictions.append('no')
    print(f't{i + 1} {label_predictions}')
