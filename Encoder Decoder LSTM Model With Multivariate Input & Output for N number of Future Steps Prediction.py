#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'.\data\stock_prices_sample.csv')
df = df[df.TICKER != 'GEF']
df = df[df.TYPE != 'Intraday']
df.reset_index(drop = True, inplace =True)


# In[3]:


TRAIN_SPLIT=700

LOOK_BACK_WINDOW = 90
NUM_OF_FUTURE_PREDICTION = 13


BATCH_SIZE = 128
EPOCHS = 10
EVALUATION_INTERVAL = 20
VALIDATION_INTERVAL = 10


# ## Data preprocessing

# In[4]:


features_considered = ["OPEN", "HIGH", "LOW", "CLOSE"]
features = df[features_considered]


# In[64]:


data = features.values
print(data.shape)
data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
dataset = (data-data_mean)/data_std


# In[6]:


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(dataset[indices])
        labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)


# In[7]:


x_train, y_train = multivariate_data(dataset=dataset
                                     , target=dataset[:,[0,3]]
                                     , start_index=0
                                     , end_index=TRAIN_SPLIT
                                     , history_size=LOOK_BACK_WINDOW
                                     , target_size=NUM_OF_FUTURE_PREDICTION)

x_val, y_val = multivariate_data(dataset=dataset
                                 , target=dataset[:,[0,3]]
                                 , start_index=TRAIN_SPLIT
                                 , end_index=None
                                 , history_size=LOOK_BACK_WINDOW
                                 , target_size=NUM_OF_FUTURE_PREDICTION)

print("shape of x_train:", x_train.shape)
print("shape of y_train:", y_train.shape)
print("shape of x_val:", x_val.shape)
print("shape of y_val:", y_val.shape)


# ### RNN LSTM Encoder & Decoder Network

# In[18]:


model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=x_train.shape[-2:]))
model.add(RepeatVector(y_train.shape[1]))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(y_train.shape[2])))

model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mse')


# ### Model Trining

# In[19]:


model_history = model.fit(x=x_train, y=y_train
                          , epochs=EPOCHS
                          , steps_per_epoch=EVALUATION_INTERVAL
                          , validation_data=(x_val, y_val)
                          , validation_steps=VALIDATION_INTERVAL)


# ### Train & CV Error (curve by epoch)

# In[21]:


def plot_train_history(history, title):
    loss = history.history['loss'][1:]
    val_loss = history.history['val_loss'][1:]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

plot_train_history(model_history, 'Multi-Step Training and validation loss')


# ### Model Summary

# In[22]:


model.summary()


# ### Prediction

# In[43]:


data_predict = dataset[-90:,:]
data_predict = data_predict.reshape((1, data_predict.shape[0], data_predict.shape[1]))
data_extrapolated = (model.predict(data_predict)[0] * data_std[[0,3]]) + data_mean[[0,3]]


# In[65]:



layout = (1,2)
plt.figure(figsize=(17, 8))
plt.subplot2grid(layout, (0,0))
plt.plot(data[:,[0,3]], label='training data model')
plt.axis([0, 995, 10, 25])
plt.legend()
plt.grid(True)
plt.title('Actual')
plt.subplot2grid(layout, (0,1))
plt.plot(data_extrapolated, label='testing data model')
plt.title('Forecast')
plt.axis([0, 13, 10, 25])
plt.legend()
plt.grid(True);

