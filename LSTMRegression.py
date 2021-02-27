# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fixing random seed for reproducibility
np.random.seed(42)

# load the dataset

df = pd.read_csv('./airline-passengers.csv')
data = df.values
data = df.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# split into train and tst sets
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size: len(data), :]
# print(len(train), len(test))

# reshape into X=t and y=t+1
look_back = 1
train_X, train_y = create_dataset(train, look_back)
test_X, test_y = create_dataset(test, look_back)

# reshape inputs to be [samples, time-steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2)

# make predictions
train_preds = model.predict(train_X)
test_preds = model.predict(test_X)

# invert predictions
train_preds = scaler.inverse_transform(train_preds)
train_y = scaler.inverse_transform([train_y])
test_preds = scaler.inverse_transform(test_preds)
test_y = scaler.inverse_transform([test_y])

# calculate root mean squared error
train_score = math.sqrt(mean_squared_error(train_y[0], train_preds[:, 0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = math.sqrt(mean_squared_error(test_y[0], test_preds[:, 0]))
print('Test Score: %.2f RMSE' % (test_score))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_preds) + look_back, :] = train_preds

# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_preds) + (look_back * 2) + 1:len(data) - 1, :] = test_preds

# plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
