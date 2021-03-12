import pyodbc 
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime as dt
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas_datareader as web
import pandas as pd
import urllib.request, json
import tensorflow as tf

"""
    Obtain data
    ----------------------------------------------------------------
"""
conn = conn = eval(os.environ.get('MALACHI_SERVER'))

cursor = conn.cursor()
cursor.execute('SELECT [DailyId], [High], [Low], [Date], [Close] FROM Stonks.dbo.[AAPL_Daily]')

l = []

query_results = []
count = 0
for row in cursor:
    tup = tuple(row)
    query_results.append(tup)
    count += 1

# Sort the results by date
query_results = sorted(query_results, key=itemgetter(0))

# Store data to DataFrame
data = pd.DataFrame(query_results, columns=['DailyId', 'High', 'Low', 'Date', 'Close'])
# print(data)

"""
    Obtain data
    ----------------------------------------------------------------
"""

"""
    Train data and scale
    ----------------------------------------------------------------
"""

# Establish train data
split = 400
train_data = data['Close'][:split].values.reshape(-1,1)


# Scale data
scaler =  MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data)

prediction_days = 60 # How many days to reference for a single prediction

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

"""
    Train data and scale
    ----------------------------------------------------------------
"""

"""
    Create and train model
    ----------------------------------------------------------------
"""

# Build the model
model = Sequential()

model.add(LSTM(units=55, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=55, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=55))
model.add(Dropout(0.2))
model.add(Dense(units=5)) # prediction of the next closing

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32)

"""
    Create and train model
    ----------------------------------------------------------------
"""


""" 
    Test the model accuracy on test data 
    ----------------------------------------------------------------
"""

# Load Test Data
test_data = data['Close'][split:]
actual_prices = test_data.values

total_dataset = pd.concat((data['Close'][:split], test_data), axis=0)

# model_inputs is a 2D list with an index of 0
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:]
# Convert this so that user can input how many days into the future they wanna look
# model_inputs.loc[len(model_inputs.index)] = 0
# model_inputs.loc[len(model_inputs.index)] = 0
# model_inputs.loc[len(model_inputs.index)] = 0
model_inputs = model_inputs.values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []
count = 0
l = len(model_inputs)

for x in range(prediction_days, l + 1):
    # [..., 0] returns the splice in the first index (0)
    # since model_inputs is a 2D array with index 0
    # appends prediction days values and creates 
    # len(total_dataset) + 1 - split rows

    if model_inputs[x-prediction_days:x, 0][-1] < 0 :
        real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict(real_data)
        prediction = prediction.tolist()

        # print('here\n')    
        # print(len(model_inputs[x-prediction_days:x, 0]))
        # print(model_inputs[x-prediction_days:x, 0])
        model_inputs[x-prediction_days:x, 0].put(59, [prediction])
        # print(len(model_inputs[x-prediction_days:x, 0]))
        # print(model_inputs[x-prediction_days:x, 0])
        # print('done here\n')

    x_test.append(model_inputs[x-prediction_days:x, 0]) 

    # print(len(x_test[-1]))
    # print(x_test[-1], count, type(model_inputs[x-prediction_days:x, 0]))
    count += 1

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # (test_data length + 1, prediction days, 1)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Predict next day
# Convert this to show the closing prices for the amount of days 
# the user chose to look ahead
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print("Prediction: {}".format(prediction))

# Plot the test predictions
plt.plot(actual_prices, color='black', label="Actual Apple price")
plt.plot(predicted_prices, color='red', label="Predicted Apple price")
plt.title("Apple Share Price")
plt.xlabel('Date')
plt.ylabel('Apple Share price')
plt.legend()
plt.show()


