#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cia
"""
import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt

FullSet_0 = pd.read_csv('./Datasets/1000ceros.csv', header=None)
FullSet_1 = pd.read_csv('./Datasets/1000unos.csv',  header=None)

#--Quick rescale to [0,1] given that any pixel is in [0,255]
FullSet_0 = (FullSet_0 / 255.0)
FullSet_1 = (FullSet_1 / 255.0)


def feat_extraction(data, theta=0.1):
    # data: dataframe
    # theta: parameter of the feature extraction
    #
    features = np.zeros([data.shape[0], 8]) #<- allocate memory with zeros
    data = data.values.reshape([data.shape[0],28,28]) 
    #-> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        #..current image 
        x = data[k,:,:]
        #--width feature
        sum_cols = x.sum(axis=0) #<- axis0 of x, not of data!!
        indc = np.argwhere(sum_cols > theta * sum_cols.max())
        col_3maxs = np.argsort(sum_cols)[-3:] 
        features[k,0] = indc[-1] - indc[0]
        features[k,1:4] = col_3maxs
        #--width feature
        sum_rows = x.sum(axis=1) #<- axis1 of x, not of data!!
        indr = np.argwhere(sum_rows > theta * sum_rows.max())
        features[k,4] = indr[-1] - indr[0]
        row_3maxs = np.argsort(sum_rows)[-3:] 
        features[k,5:8] = row_3maxs
    col_names = ['width','W_max1','W_max2','W_max3','height','H_max1','H_max2','H_max3']
    return pd.DataFrame(features,columns = col_names)    


theta = 0.5
FeatSet_0 = feat_extraction(FullSet_0)
FeatSet_1 = feat_extraction(FullSet_1)



def feat_area(features):
    area = features['width'] * features['height']
    return pd.DataFrame({'area':area})

area=feat_area(FeatSet_0)
FeatSet_0 = pd.concat([FeatSet_0, area],axis=1)

area=feat_area(FeatSet_1)
FeatSet_1 = pd.concat([FeatSet_1, area],axis=1)


def jitter(X,sigma=0.3):
    random_sign = (-1)**np.random.randint(1,3,*X.shape)
    return X + np.random.normal(0,sigma,*X.shape)*random_sign


horizontal = 'H_max1'
vertical = 'area'
alpha = 0.1
sigma = 0.3
plt.plot(jitter(FeatSet_0[horizontal],sigma), jitter(FeatSet_0[vertical],sigma),'yo',alpha=alpha)
plt.plot(jitter(FeatSet_1[horizontal],sigma), jitter(FeatSet_1[vertical],sigma),'bx',alpha=alpha)
plt.xlabel(horizontal)
plt.ylabel(vertical)
plt.title('0(yellow) vs.1(blue),  with theta=%0.2f'%theta)
plt.show()     