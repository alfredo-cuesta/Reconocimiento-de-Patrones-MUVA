#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cia
"""
#%%
import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt

#%%
# --- Read 1000ceros.csv + 1000unos.csv  -------------------------------------

FullSet_0 = np.genfromtxt('./Datasets/1000ceros.csv', delimiter=',') #<--- 
FullSet_1 = np.genfromtxt('./Datasets/1000unos.csv', delimiter=',')  #<---

# --- Print out the shape of these two datasets
print("Shape of FullSet_0 is %d x %d" 
      % (FullSet_0.shape[0], FullSet_0.shape[1]) ) 
print("Shape of FullSet_1 is %d x %d" 
      % (FullSet_1.shape[0], FullSet_1.shape[1]) )  

# --- Visualize the datasets
f1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(FullSet_0,cmap='Blues')
ax2.imshow(FullSet_1,cmap='Blues')

#%%

instance_ID_to_show = 232 #<-- pick a number in [0,999]

# --- Plot an instance of ZERO
f2, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
instance_0 = FullSet_0[instance_ID_to_show,:]                        #<---
ax1.imshow(np.reshape(instance_0,[28,28]), 'Blues')

# --- Plot an instance of ONE
instance_1 = FullSet_1[instance_ID_to_show,:]                        #<---
ax2.imshow(np.reshape(instance_1,[28,28]), 'Blues')


plt.show(f1)
plt.show(f2)

