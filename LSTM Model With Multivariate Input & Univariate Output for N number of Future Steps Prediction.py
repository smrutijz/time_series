#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'.\data\stock_prices_sample.csv')
df = df[df.TICKER != 'GEF']
df = df[df.TYPE != 'Intraday']
df.reset_index(drop = True, inplace =True)


# In[3]:


TRAIN_SPLIT=700

features_considered = ["CLOSE"]
LOOK_BACK = 90
NUM_OF_FUTURE_PREDICTION = 1


BATCH_SIZE = 128
EPOCHS = 300
EVALUATION_INTERVAL = 20
VALIDATION_INTERVAL = 10


# ## Data preprocessing

# In[4]:


features = df[features_considered]
data = features.values


# In[5]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)


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
                                     , target=dataset[:, 0]
                                     , start_index=0
                                     , end_index=TRAIN_SPLIT
                                     , history_size=LOOK_BACK
                                     , target_size=NUM_OF_FUTURE_PREDICTION)

x_val, y_val = multivariate_data(dataset=dataset
                                 , target=dataset[:, 0]
                                 , start_index=TRAIN_SPLIT
                                 , end_index=None
                                 , history_size=LOOK_BACK
                                 , target_size=NUM_OF_FUTURE_PREDICTION)

print("shape of x_train:", x_train.shape)
print("shape of y_train:", y_train.shape)
print("shape of x_val:", x_val.shape)
print("shape of y_val:", y_val.shape)


# In[8]:


model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=x_train.shape[-2:]))
model.add(LSTM(16, activation='relu'))
model.add(Dense(NUM_OF_FUTURE_PREDICTION))

model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mean_squared_error')


# In[9]:


model_history = model.fit(x=x_train, y=y_train
                          , epochs=EPOCHS
                          , steps_per_epoch=EVALUATION_INTERVAL
                          , validation_data=(x_val, y_val)
                          , validation_steps=VALIDATION_INTERVAL)


# ### Train & cv error curve by epoch

# In[10]:


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

plot_train_history(model_history, 'Multi-Step Training and validation loss')


# ### Visualizing

# In[11]:


# make predictions
trainPredict = model.predict(x_train)
cvPredict = model.predict(x_val)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
cvPredict = scaler.inverse_transform(cvPredict)

trainY = scaler.inverse_transform(y_train)
cvY = scaler.inverse_transform(y_val)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if NUM_OF_FUTURE_PREDICTION == 1:
    trainError = mean_absolute_percentage_error(trainPredict, trainY)
    testError = mean_absolute_percentage_error(cvPredict, cvY)
    layout = (1,2)
    plt.figure(figsize=(17, 8))
    plt.subplot2grid(layout, (0,0))
    plt.plot(trainY, color='b', label='training data actual')
    plt.plot(trainPredict , color='g', label='training data model')
    plt.legend()
    plt.grid(True)
    plt.title('Mean Absolute Percentage Training Error: {0:.2f}%'.format(trainError))
    plt.subplot2grid(layout, (0,1))
    plt.plot(cvY, color='b', label='testing data actual')
    plt.plot(cvPredict , color='r', label='testing data model')
    plt.title('Mean Absolute Percentage Testing Error: {0:.2f}%'.format(testError))
    plt.legend()
    plt.grid(True);
    plt.tight_layout()
else:
    trainError = mean_absolute_percentage_error(trainPredict[-1], trainY[-1])
    testError = mean_absolute_percentage_error(cvPredict[-1], cvY[-1])
    layout = (1,2)
    plt.figure(figsize=(17, 8))
    plt.subplot2grid(layout, (0,0))
    plt.plot(trainY[-1], color='b', label='training data actual')
    plt.plot(trainPredict[-1] , color='g', label='training data model')
    plt.legend()
    plt.grid(True)
    plt.title('Mean Absolute Percentage Training Error: {0:.2f}%'.format(trainError))
    plt.subplot2grid(layout, (0,1))
    plt.plot(cvY[-1], color='b', label='testing data actual')
    plt.plot(cvPredict[-1] , color='r', label='testing data model')
    plt.title('Mean Absolute Percentage Testing Error: {0:.2f}%'.format(testError))
    plt.legend()
    plt.grid(True);
    plt.tight_layout()


# In[12]:


model.summary()

