#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cia
"""

import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt

#%%

FullSet_0 = pd.read_csv('./Datasets/1000ceros.csv', header=None)
FullSet_1 = pd.read_csv('./Datasets/1000unos.csv',  header=None)

#--Quick rescale to [0,1] given that any pixel is in [0,255]
FullSet_0 = (FullSet_0 / 255.0)
FullSet_1 = (FullSet_1 / 255.0)

#%%

theta = 0.5  #<- feature parameter
instance_ID_to_show = 123 #<-- pick a number in [0,999] 

x = FullSet_0.iloc[instance_ID_to_show].values.reshape([28,28])

#--width feature
sum_cols = x.sum(axis=0)
indc = np.argwhere(sum_cols > theta * sum_cols.max())
width = indc[-1] - indc[0]

#--height feature
sum_rows = x.sum(axis=1)
indr = np.argwhere(sum_rows > theta * sum_rows.max())
height = indr[-1] - indr[0]
features_df = pd.DataFrame({'width':width, 'height':height})

##------------ we can visualize and print the two features 
plt.rcParams['figure.figsize'] = [7,7] #<--makes the figures larger in Jupyter
# show "width" 
plt.subplot(221)
plt.plot(sum_cols)
plt.plot(indc[[0,-1]],sum_cols[indc[[0,0]]],'r')
plt.title('columns -> width')
# show "height" 
plt.subplot(222)
plt.plot(sum_rows)
plt.plot(indr[[0,-1]],sum_rows[indr[[0,0]]],'m')
plt.title('rows -> height')
# show image in the "input space" 
plt.subplot(223)
plt.imshow(x,cmap='Blues')
plt.plot(indc[[0,-1]],sum_cols[indc[[0,0]]],'r')
plt.plot(sum_rows[indr[[0,0]]],indr[[0,-1]],'m')
plt.title('Input space')
# show image in the "feature space" = show the point (width, height)
plt.subplot(224)
plt.plot(width,height,'o')
plt.axis([0,30,0,30])
plt.title('Feature space')
plt.show()
# --- Print out the features of the image
print("The image in the feature space is (%d, %d)" 
      % (features_df['width'], features_df['height'])  )
