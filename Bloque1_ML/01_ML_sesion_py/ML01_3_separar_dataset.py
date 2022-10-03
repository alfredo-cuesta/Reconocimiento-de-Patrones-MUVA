#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cia
"""

import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt

#%%

def shuffle_split_2(data, ratio):
    # data is a dataframe
    # ratio in [0,1], is the percentage of the data for the 2nd split
    # we call the 1st split "train_set" and the second split "test_set"
    
    # 1) shuffle the rows of the dataframe
    shuffled_indices = np.random.permutation(data.shape[0])
    # 2) get the indices for each split
    test_set_rows = int(data.shape[0] * ratio)
    test_indices = shuffled_indices[:test_set_rows]
    train_indices = shuffled_indices[test_set_rows:]
    # 3) get the two splits
    train_set = data.iloc[train_indices]
    test_set  = data.iloc[test_indices]
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)
    
#%%
fraction_Test = 0.2      #<- Percentage of the dataset held for test, in [0,1]
fraction_Valid= 0.2      #<- Percentage of the train set held for validation, in [0,1]
    
np.random.seed(seed=123) #<- to force that every run produces the same outcome
                         #    (comment, or remove, to get randomness)
                         
# --- Get data -------------------------------------
FullSet_0 = pd.read_csv('./Datasets/1000ceros.csv', header=None)
FullSet_1 = pd.read_csv('./Datasets/1000unos.csv',  header=None)

# --- Separate Test set -----------------------------
TrainSet_0, TestSet_0 = shuffle_split_2(FullSet_0, fraction_Test)
TrainSet_1, TestSet_1 = shuffle_split_2(FullSet_1, fraction_Test)

# --- Separate Validation set -----------------------
TrainSet_0, ValidSet_0 = shuffle_split_2(TrainSet_0, fraction_Valid)
TrainSet_1, ValidSet_1 = shuffle_split_2(TrainSet_1, fraction_Valid)

# --- Print out the shape of all these datasets
print("Shape of FullSet_0 is %d x %d" 
      % (FullSet_0.shape[0], FullSet_0.shape[1]) ) 
print("Shape of FullSet_1 is %d x %d" 
      % (FullSet_1.shape[0], FullSet_1.shape[1]) )  

print("Shape of TrainSet_0 is %d x %d" 
      % (TrainSet_0.shape[0], FullSet_0.shape[1]) ) 
print("Shape of TrainSet_1 is %d x %d" 
      % (TrainSet_1.shape[0], FullSet_1.shape[1]) )  

print("Shape of ValidSet_0 is %d x %d" 
      % (ValidSet_0.shape[0], FullSet_0.shape[1]) ) 
print("Shape of ValidSet_1 is %d x %d" 
      % (ValidSet_1.shape[0], FullSet_1.shape[1]) )  

print("Shape of TestSet_0 is %d x %d" 
      % (TestSet_0.shape[0], FullSet_0.shape[1]) ) 
print("Shape of TestSet_1 is %d x %d" 
      % (TestSet_1.shape[0], FullSet_1.shape[1]) )  
