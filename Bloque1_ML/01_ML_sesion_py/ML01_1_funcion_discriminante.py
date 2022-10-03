#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
´Funcion discriminante y superficie de decision
@author: cia
"""
#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
def decision_surface(x1, x2):
    return -3*x1 + 1*x2


Ns = 100

x1lin = np.linspace(-10, 10, Ns)
x2lin = np.linspace(-10, 10, Ns)
x1v, x2v = np.meshgrid(x1lin, x1lin)
f = decision_surface(x1v, x2v)

# Plot the surface.

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x1v, x2v, f, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=9)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')

plt.show()

#%%

def discriminant_function(x1, x2):
    z = decision_surface(x1, x2)
    z[z>0]=1
    z[z<0]=-1
    return z
    
z = discriminant_function(x1v,x2v)

plt.imshow(z, extent=[x1lin[0], x1lin[-1], x2lin[0], x2lin[-1] ])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()

plt.show()
