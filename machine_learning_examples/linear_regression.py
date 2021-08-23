import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
data = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv', header=None).values
print(data.shape)
#%%
X = data[:,0].reshape(-1,1)
y = data[:,1]
plt.scatter(X, y)

#%%
y = np.log(y)
plt.scatter(X, y)

#%%

X = X - X.mean()

#%%

model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(1,)),
                                   tf.keras.layers.Dense(1)])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')

def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

res = model.fit(X, y, epochs=200, callbacks=[scheduler])
#%%

plt.plot(res.history['loss'], label='loss')
#%%

print(model.layers[0].get_weights())
a = model.layers[0].get_weights()[0][0,0]

y_hat = model.predict(X).flatten()
plt.scatter(X, y)
plt.plot(X, y_hat)

#%%
print(model.predict(X).flatten())




