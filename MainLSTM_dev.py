# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
import os.path
from os import path
# Training params
EPOCHS = 20
NUM_UNITS = 64
BATCH_SIZE = 64
dropout_rate = 0.2

# Settings
FDAYS = 120 # Financial days

# Importing the training set
name_stock = 'A2A.MI'
model_name = "SPP_"+name_stock+"_Weights.h5"
dataset = pd.read_csv(name_stock + '.csv')
dataset.fillna(dataset.mean(), inplace=True)
dataset = dataset.iloc[:, 1:2].values
# Training data until 3 months ago
training_set = dataset[:-FDAYS,:]

# Testing data is 3 previous months plus last 3 months to predict
testing_set = np.concatenate((training_set[-FDAYS:,:], dataset[-FDAYS:,:]))

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(testing_set)

# TRAINING SET: Creating a data structure with FDAYS timesteps and 1 output
X_train = []
y_train = []
for i in range(FDAYS, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-FDAYS:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# TESTING SET: Creating a data structure with FDAYS timesteps and 1 output
X_test = []
for i in range(FDAYS, testing_set_scaled.shape[0]):
    X_test.append(testing_set_scaled[i-FDAYS:i, 0])
X_test = np.array(X_test)
# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = NUM_UNITS, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(dropout_rate))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = NUM_UNITS, return_sequences = True))
regressor.add(Dropout(dropout_rate))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = NUM_UNITS, return_sequences = True))
regressor.add(Dropout(dropout_rate))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = NUM_UNITS, return_sequences = True))
regressor.add(Dropout(dropout_rate))

# Adding a fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = NUM_UNITS))
regressor.add(Dropout(dropout_rate))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# load weights into new model
if path.exists(model_name):
    regressor.load_weights(model_name)
    print("Loaded model from disk")
else:
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

# serialize weights to HDF5
regressor.save_weights(model_name)
print("<Info> Model "+model_name+" saved to disk.")


# Making the predictions and visualising the results
predicted_stock_price_1day = regressor.predict(X_test)
predicted_stock_price_1day = sc.inverse_transform(predicted_stock_price_1day)

# Visualising the results
plt.plot(testing_set[-FDAYS:], color = 'red', label = 'Real '+name_stock)
plt.plot(predicted_stock_price_1day, '<', color = 'blue', label = 'Predicted 1-day '+name_stock)
plt.title(name_stock + ' Prediction')
plt.title(name_stock + ' Prediction')
plt.xlabel('Time')
plt.ylabel(name_stock + ' Price')
plt.legend()
plt.grid()
plt.show()

print('<Info> Process terminated.')