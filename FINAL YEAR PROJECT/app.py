from flask import Flask, render_template, request, after_this_request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from models2 import train_lstm_model, train_linear_regression_model,closeplot, predict_next_day_lstm, predict_next_day_regression, global_scaler
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objects as go
import plotly.io as pio


import warnings
warnings.filterwarnings('ignore')

app = Flask("Stock Price App")

# Global variables to store the trained models
trained_lstm_model = None
trained_linear_regression_model = None
accuracyPercentage = None
accuracyPercentage1 = None

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/contact_us')
def contact_us():
   return render_template('contact_us.html')

@app.route('/team')
def team():
   return render_template('team.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == "POST":
        try:
            file = request.files['file']
            df = pd.read_csv(file, na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
            # Handle missing values here
            df.fillna(df.mean(), inplace=True)  # Replace missing values with mean
            
            closeplot(df)
            output_var = pd.DataFrame(df['Close'])
            features = ['Open', 'High', 'Low', 'Volume']
            scaler = MinMaxScaler(feature_range=(0, 1))
            feature_transform = scaler.fit_transform(df[features])
            feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)
            timesplit = TimeSeriesSplit(n_splits=10)
            
            for train_index, test_index in timesplit.split(feature_transform):
                X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index) + len(test_index))]
                y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index) + len(test_index))].values.ravel()
            
            trainX = np.array(X_train)
            testX = np.array(X_test)
            X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1]) #The `reshape()` function changes the shape of the array to `(number of samples, time steps, number of features)`
            X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

            global trained_lstm_model,accuracyPercentage
            trained_lstm_model,accuracyPercentage = train_lstm_model(X_train, X_test, y_train, y_test, df, train_index, test_index)

            global trained_linear_regression_model,accuracyPercentage1
            trained_linear_regression_model,accuracyPercentage1 = train_linear_regression_model(X_train, X_test, y_train, y_test, df, train_index, test_index)
            
            global global_scaler
            global_scaler.fit(df[features].values)
            
            # Render the HTML content for file-upload-message
            message = "File uploaded and processed"
            return message
        except Exception as e:
            message = "File error"
            return message

    else:
        # Set show_spinner initially to True
        show_spinner = True
        return render_template('train.html', show_spinner=show_spinner)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the POST request
        open_price = float(request.form.get('open_price'))
        high_price = float(request.form.get('high_price'))
        low_price = float(request.form.get('low_price'))
        volume = float(request.form.get('volume'))

        
        # Use the trained models to make predictions
        next_day_price_lstm = predict_next_day_lstm(trained_lstm_model, open_price, high_price, low_price, volume)
        next_day_price_regression = predict_next_day_regression(trained_linear_regression_model, open_price, high_price, low_price, volume)


        # Extract the single floating-point number from the array
        next_day_price_lstm_float = next_day_price_lstm[0][0]
        next_day_price_regression_float = next_day_price_regression[0][0]

        # Convert the numbers to strings
        next_day_price_lstm_str = str(next_day_price_lstm_float)
        next_day_price_regression_str = str(next_day_price_regression_float)

        # Pass the predictions to the template
        return render_template('predict.html', lstm_prediction=next_day_price_lstm_str, regression_prediction=next_day_price_regression_str,acc1=accuracyPercentage,acc2=accuracyPercentage1)
    else:
        return render_template('predict.html')

  
if __name__ == '__main__':
    debug = False
    # debug = True
    port = 1200
    #app.run(debug=True)
    app.run(
        debug=debug,
        port=port
    )