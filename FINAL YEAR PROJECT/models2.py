# models.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM,Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
###########
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#########################
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from tensorflow.keras import regularizers # type: ignore

import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import MinMaxScaler
# Global variable to store the fitted scaler
global_scaler = MinMaxScaler()


import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

def closeplot(df):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['Close'], linestyle='-')
    ax.set_title('Closing prices over time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    plotly_fig = tls.mpl_to_plotly(fig)
    pio.write_html(plotly_fig, file="static/plots/Close.html")
    #close plot




def train_lstm_model(X_train, X_test, y_train, y_test,df,train_index,test_index):
    global global_scaler
    features = ['Open', 'High', 'Low', 'Volume']
    global_scaler.fit(df[features].values)
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    """ # Generate the plot
    plt.figure()    #create a new figure
    plt.plot(df.index[len(train_index): (len(train_index)+len(test_index))], y_test, label='Actual', linestyle='-')
    plt.plot(df.index[len(train_index): (len(train_index)+len(test_index))], predictions, label='Predicted', linestyle='-')
    plt.title('Stock Market Prediction - LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig('static/images/lstm.png')"""
    # Calculate accuracy
    accuracy = 0
    for i in range(len(y_test)):
        accuracy += (abs(y_test[i] - predictions[i])/y_test[i])*100
    accuracyPercentage = 100 - accuracy/len(y_test)

    print(f"Accuracy for LSTM model: {accuracyPercentage}%")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(df.index[len(train_index): (len(train_index)+len(test_index))], y_test, label='Actual', linestyle='-')
    ax.plot(df.index[len(train_index): (len(train_index)+len(test_index))], predictions, label='Predicted', linestyle='-')
    ax.set_title('Stock Market Prediction - LSTM')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()

    # Convert matplotlib figure to plotly
    plotly_fig = tls.mpl_to_plotly(fig)
    # Display plotly figure
    #iplot(plotly_fig)
    # Save plotly figure as an HTML file
    #pio.write_html(plotly_fig, file='output.html', auto_open=True)
    pio.write_html(plotly_fig, file="static/plots/lstm_plot.html")
    return model,accuracyPercentage
    
    
def train_linear_regression_model(X_train, X_test, y_train, y_test,df,train_index,test_index):
    global global_scaler
    # Assuming 'features' is a list of column names for the features
    features = ['Open', 'High', 'Low', 'Volume']
    # Fit the scaler with the training data
    global_scaler.fit(df[features].values)
    # Build the linear regression model
    model1 = Sequential()   #fully connected dense layer and dim as number of samples in each row
    model1.add(Dense(64, input_dim=X_train.shape[2], activation='relu'))    #telling the layer that each input sample should have X_train.shape[2] features. 
    model1.add(Dropout(0.4))#This is crucial for the layer to know how many input units it should expect for each sample. 
    model1.add(Dense(32, activation='relu'))  # Second hidden layer with 32 neurons and ReLU activation
    model1.add(Dropout(0.5))
    model1.add(Dense(1))  # Output layer for regression prediction
    # Compile the model
    model1.compile(optimizer=Adam(learning_rate=0.05), loss='mean_squared_error')
    #Adam (Adaptive Moment Estimation) that adapts the learning rate which adjusts weights for each neuron
    # Reshape input data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

    # Train the model
    model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Make predictions on the test set
    predictions1 = model1.predict(X_test)
    # Calculate accuracy
    accuracy = 0
    for i in range(len(y_test)):
        accuracy += (abs(y_test[i] - predictions1[i])/y_test[i])*100
    accuracyPercentage = 100 - accuracy/len(y_test)

    print(f"Accuracy for Linear Regression model: {accuracyPercentage}%")

    """ # Generate the plot
    plt.figure()    #create a new figure
    plt.plot(newdf.index[len(train_index): (len(train_index)+len(test_index))], y_test, label='Actual', linestyle='-')
    plt.plot(newdf.index[len(train_index): (len(train_index)+len(test_index))], predictions1, label='Predicted', linestyle='-')
    plt.title('Stock Market Prediction - Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig('static/images/linearg.png')"""

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(df.index[len(train_index): (len(train_index)+len(test_index))], y_test, label='Actual', linestyle='-')
    ax.plot(df.index[len(train_index): (len(train_index)+len(test_index))], predictions1, label='Predicted', linestyle='-')
    ax.set_title('Stock Market Prediction - Linear Regression')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()

    # Convert matplotlib figure to plotly
    plotly_fig = tls.mpl_to_plotly(fig)
    # Display plotly figure
    #iplot(plotly_fig)
    # Save plotly figure as an HTML file
    #pio.write_html(plotly_fig, file='output.html', auto_open=True)
    pio.write_html(plotly_fig, file="static/plots/linear_plot.html")
    return model1,accuracyPercentage






def predict_next_day_lstm(model, open_price, high_price, low_price, volume):
    global global_scaler
    # Scale the input data
    input_data = global_scaler.transform([[open_price, high_price, low_price, volume]])
    # Reshape the input data for the LSTM model
    input_data = input_data.reshape(1, 1, 4) #It reshapes the input data to have dimensions `(1, 1, 4)`, where 1 is the number of samples, 1 is the number of time steps, and 4 is the number of features.
    # Make a prediction for the next day's closing price
    next_day_prediction = model.predict(input_data)
    return next_day_prediction

"""def predict_next_day_regression(model1, open_price, high_price, low_price, volume):
    global global_scaler
    # Scale the input data
    input_data = global_scaler.transform([[open_price, high_price, low_price, volume]])
    # Reshape the input data for the LSTM model
    input_data = input_data.reshape(1, 1, 4)
    # Make a prediction for the next day's closing price
    next_day_prediction1 = model1.predict(input_data)
    return next_day_prediction1"""

def predict_next_day_regression(model1, open_price, high_price, low_price, volume):
    global global_scaler
    input_data = np.array([[open_price, high_price, low_price, volume]])
    input_data = global_scaler.transform(input_data)
    input_data = input_data.reshape(1, 4)  # Reshape the input data to match the model's input shape
    next_day_prediction1 = model1.predict(input_data)
    return next_day_prediction1