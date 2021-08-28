import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

N = 1000 
X = np.random.random((N, 2))*6-3 #uniformly distributed between (-3,3)
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

# y = cos(2*x1) + cos(3*x2)

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], Y)
plt.show()
