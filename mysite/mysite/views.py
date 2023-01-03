"Ibrahim Aminu"

from django.shortcuts import render
from django.contrib import auth
import pyrebase
from django.shortcuts import redirect

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

#
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#
#
from tensorflow.python.keras.layers import LSTM
#

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#
#
import datetime as dt


from django.http import  HttpResponse
from django.shortcuts import render
from django.contrib import  auth
import pyrebase
import  sklearn

# arr = [2,"four ", "six", 8]
# def hello_world(request):
#
#     return  HttpResponse("Hello World we got this! " )


config = {

    'apiKey': "AIzaSyAWuD0RF1NXotnqieNh-FkFhf730lQ0PAc",

    'authDomain': "sd3-group-project.firebaseapp.com",

    'projectId': "sd3-group-project",

    'storageBucket': "sd3-group-project.appspot.com",

    'messagingSenderId': "918599092511",

    'appId': "1:918599092511:web:93712ecd7113de0c56c82a",

    'databaseURL': "https://sd3-group-project-default-rtdb.europe-west1.firebasedatabase.app/"

}
firebase = pyrebase.initialize_app(config)

authent = firebase.auth()

database = firebase.database()



def viewclient(request):
    all_users = database.child("users").child("fmNKSTflBTfaghFzhvo5pDQdTbM2").child("clients").get()
    list = []

    for user in all_users.each():
                # print(user.key())
                print(user.val())  # {name": "Mortimer 'Morty' Smith"}
                list.append(user.val())
    print(list)




    return render(request,"welcome.html", {'list': list})


def signIn(request):

    return render(request,"signIn.html")

def predictions(request):

    return render(request,"predictions.html")




def postsign(request):
    email = request.POST.get('email')
    passw = request.POST.get("pass")
    print(email,passw)

    try:
        user = authent.sign_in_with_email_and_password(email,passw)
    except:
        message="Invalid Credentials"
        return render(request,"signIn.html",{"m":message})

    print(user['idToken'])
    # Creating a session token
    session_id = user['idToken']
    request.session['uid'] = str(session_id)

    return render(request,"welcome.html",{"e":email})

def postregister(request):
    name = request.POST.get('name')
    email = request.POST.get('email')
    passw = request.POST.get("pass")
    print(email,passw)

    # Creating Account

    try:
         user = authent.create_user_with_email_and_password(email,passw)
    # Getting Unique user id
    except:
        message = "Account could not be created try again!"
        return render(request,"register.html",{"m":message})

    uid = user['localId']

    data = {"name":name,"status":"1"}

    database.child("users").child(uid).child("details").set(data)



    return render(request,"signIn.html")

def home(request):
    return render(request,"home.html")


def logout(request):
    auth.logout(request)
    return render(request,'signIn.html')

def register(request):
    return render(request,"register.html")


def stockPredict():

    nvda_data = pd.read_csv('C:/Users/Ibrahim/Documents/GroupProject/mysite/mysite/csv/AAPL.csv',index_col='Date')
    nvda_data.head()
    #Open - Price When the market Opens
    #High - Highest recorded price for the day
    #Low - Lowest recorded price for the day
    #Close - Price when the market closes
    #Adj Close - Modified closing price based on corporate actions
    #Volume - Amount of stocks sold in a day

    # plt.figure(figsize=(15,10))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
    # x_dates = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in nvda_data.index.values]

    # plt.plot(x_dates,nvda_data['Open'],label = 'Open')
    # plt.plot(x_dates,nvda_data['Close'], label = 'Close')
    # plt.xlabel('Time Scale')
    # plt.ylabel('Scaled USD')
    # plt.legend()
    # plt.gcf().autofmt_xdate()
    # plt.show()

    target_Y = nvda_data['Close']
    X_feat = nvda_data.iloc[:,0:4]
    print(X_feat)
#     Feature Scaling
    sc = StandardScaler()
    X_ft = sc.fit_transform(X_feat.values)
    X_ft = pd.DataFrame(columns=X_feat.columns,
                        data=X_ft,
                        index=X_feat.index)

    nvda_data_ft = X_ft




    # # Original
    # def lstm_split(data,n_steps):
    #     X,y = [],[]
    #     for i in range(len(data)-n_steps+1):
    #         X.append(data[i:i +n_steps,:-1])
    #         y.append(data[i + n_steps-1, -1])
    #
    #     return np.array(X),np.array(y)

    def lstm_split(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps + 1):
            X.append(data[i:i + n_steps, :])
            y.append(data[i + n_steps - 1, -1])

        return np.array(X), np.array(y)

    X1, y1 = lstm_split(nvda_data_ft.values, n_steps=2)



    train_split = 0.8
    split_idx = int(np.ceil(len(X1)*train_split))
    date_index = nvda_data_ft.index

    X_train, X_test = X1[:split_idx],X1[split_idx:]
    y_train, y_test = y1[:split_idx],y1[split_idx:]



    X_train_date,X_test_date =date_index[:split_idx],date_index[split_idx:]

    print(X1.shape,X_train.shape,X_test.shape,y_test.shape)

    lstm = Sequential()

    lstm.add(LSTM(32,input_shape=(X_train.shape[1],X_train.shape[2]),
                  activation='relu',return_sequences=True
                  ))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error',optimizer='adam')
    lstm.summary()

    history = lstm.fit(X_train,y_train,
                        epochs=100,batch_size=4,
                       verbose=2,shuffle=False)
    # print(history)

    # loss_values = history.history['loss']
    #
    # # Extract the epoch numbers from the history object
    # epochs = range(1, len(loss_values) + 1)
    #
    # # Plot the loss values against the epoch numbers
    # plt.plot(epochs, loss_values)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # Generate predictions for the test data

    predictions = lstm.predict(X_test)














    # Compare predictions to actual 'Close' values

    # mse = mean_squared_error(y_test, y_pred)
    predictions = predictions.flatten()

    predictions = predictions[:len(y_test)]

    # Slice the test date and test values to the same length
    X_test_date_sliced = X_test_date[:len(y_test)]

    y_test_sliced = y_test[:len(X_test_date)]
    # y_test_sliced.reshape(-1)
    # Plot the true values and the predictions
    X_test_date_sliced = [dt.datetime.strptime(d, '%d/%m/%Y').date() for d in X_test_date_sliced]

    mse = mean_squared_error(y_test_sliced, predictions)

    # Print the MSE
    print(f"Mean Squared Error: {mse}")

    # for i in range(len(predictions)):
    #     print(f"Predicted: {predictions[i]}, Actual: {y_test[y_test_sliced]}")
    #
    # mse = mean_squared_error(y_test_sliced, predictions)
    # print(f"Mean Squared Error: {mse}")

    # Set the interval and formatter for the x-axis
    interval = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b %Y")

    plt.figure(figsize=(15, 10))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(interval)

    # Rotate the tick labels at an angle
    plt.xticks(rotation=45)

    # Use a smaller font size for the tick labels
    plt.xticks(fontsize=8)

    # Plot the true values and the predictions
    plt.plot(X_test_date_sliced, y_test_sliced, label='True values')
    plt.plot(X_test_date_sliced, predictions, label='Predictions')

    plt.xlabel('Time')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.title("AAPL")
    # plt.savefig("C:/Users/Ibrahim/Documents/GroupProject/mysite/templates/static/AAPL.jpg")
    plt.show()


    #
    # plt.plot(X_test_date_sliced, y_test_sliced,  label='True values')
    # plt.plot(X_test_date_sliced, predictions,  label='Predictions')
    # plt.xlabel('Time')
    # plt.ylabel('Scaled USD')
    #
    # plt.legend()

    # plt.show()
#
# stockPredict()




def cryptoPredict():
    nvda_data = pd.read_csv('C:/Users/Ibrahim/Documents/GroupProject/mysite/mysite/csv/Bitcoin_Historical_Data.csv', index_col='Date')
    nvda_data.head()
    # Open - Price When the market Opens
    # High - Highest recorded price for the day
    # Low - Lowest recorded price for the day
    # Close - Price when the market closes
    # Adj Close - Modified closing price based on corporate actions
    # Volume - Amount of stocks sold in a day

    # plt.figure(figsize=(15,10))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
    # x_dates = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in nvda_data.index.values]

    # plt.plot(x_dates,nvda_data['Open'],label = 'Open')
    # plt.plot(x_dates,nvda_data['Close'], label = 'Close')
    # plt.xlabel('Time Scale')
    # plt.ylabel('Scaled USD')
    # plt.legend()
    # plt.gcf().autofmt_xdate()
    # plt.show()

    target_Y = nvda_data['Close']
    X_feat = nvda_data.iloc[:, 0:4]
    print(X_feat)
    #     Feature Scaling
    sc = StandardScaler()
    X_ft = sc.fit_transform(X_feat.values)
    X_ft = pd.DataFrame(columns=X_feat.columns,
                        data=X_ft,
                        index=X_feat.index)

    nvda_data_ft = X_ft

    # # Original
    # def lstm_split(data,n_steps):
    #     X,y = [],[]
    #     for i in range(len(data)-n_steps+1):
    #         X.append(data[i:i +n_steps,:-1])
    #         y.append(data[i + n_steps-1, -1])
    #
    #     return np.array(X),np.array(y)

    def lstm_split(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps + 1):
            X.append(data[i:i + n_steps, :])
            y.append(data[i + n_steps - 1, -1])

        return np.array(X), np.array(y)

    X1, y1 = lstm_split(nvda_data_ft.values, n_steps=2)

    train_split = 0.8
    split_idx = int(np.ceil(len(X1) * train_split))
    date_index = nvda_data_ft.index

    X_train, X_test = X1[:split_idx], X1[split_idx:]
    y_train, y_test = y1[:split_idx], y1[split_idx:]

    X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

    print(X1.shape, X_train.shape, X_test.shape, y_test.shape)

    lstm = Sequential()

    lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]),
                  activation='relu', return_sequences=True
                  ))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    lstm.summary()

    history = lstm.fit(X_train, y_train,
                       epochs=100, batch_size=4,
                       verbose=2, shuffle=False)
    # print(history)

    # loss_values = history.history['loss']
    #
    # # Extract the epoch numbers from the history object
    # epochs = range(1, len(loss_values) + 1)
    #
    # # Plot the loss values against the epoch numbers
    # plt.plot(epochs, loss_values)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # Generate predictions for the test data

    predictions = lstm.predict(X_test)

    threshold = 3.2
    dates = [dt.datetime.strptime(d, '%d/%m/%Y').date() for d in X_test_date]

    for i in range(len(predictions)):
        for i in range(len(predictions)):
            if np.any(predictions[i] > threshold):
                # Do something
                print("ALERT: Predicted price exceeds threshold at time step SELL AT ".upper(), dates[i])
            elif np.any(predictions[i] < threshold):
                # Do something else
                print("ALERT: Predicted price is below threshold at time step BUY AT".upper(), dates[i])

    # Compare predictions to actual 'Close' values

    # mse = mean_squared_error(y_test, y_pred)
    predictions = predictions.flatten()

    predictions = predictions[:len(y_test)]

    # Slice the test date and test values to the same length
    X_test_date_sliced = X_test_date[:len(y_test)]

    y_test_sliced = y_test[:len(X_test_date)]
    # y_test_sliced.reshape(-1)
    # Plot the true values and the predictions
    X_test_date_sliced = [dt.datetime.strptime(d, '%d/%m/%Y').date() for d in X_test_date_sliced]

    mse = mean_squared_error(y_test_sliced, predictions)

    # Print the MSE
    print(f"Mean Squared Error: {mse}")

    # for i in range(len(predictions)):
    #     print(f"Predicted: {predictions[i]}, Actual: {y_test[y_test_sliced]}")
    #
    # mse = mean_squared_error(y_test_sliced, predictions)
    # print(f"Mean Squared Error: {mse}")

    # Set the interval and formatter for the x-axis
    interval = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b %Y")

    # plt.figure(figsize=(15, 10),facecolor='#0c1c23')
    # # plt.figure(figsize=(15, 10)).set_facecolor('#0c1c23')
    # plt.grid(linestyle='--')
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.gca().xaxis.set_major_locator(interval)
    #
    # # Rotate the tick labels at an angle
    # plt.xticks(rotation=45)
    #
    # # Use a smaller font size for the tick labels
    # plt.xticks(fontsize=8)
    #
    # # Plot the true values and the predictions
    # plt.plot(X_test_date_sliced, y_test_sliced, label='True values')
    # plt.plot(X_test_date_sliced, predictions, label='Predictions')
    # plt.title("Bitcoin")
    # plt.xlabel('Time')
    # plt.ylabel('Scaled USD')
    # plt.legend()
    # # plt.savefig("C:/Users/Ibrahim/Documents/GroupProject/mysite/templates/static/btc.jpg")
    # plt.rcParams['figure.facecolor'] = '#0c1c23'
    # plt.show()
    fig, ax = plt.subplots(figsize=(15, 10))

    # Set the background color of the plot
    fig.set_facecolor('#0c1c23')
    ax.set_facecolor('#0c1c23')

    # Set the grid style and other plot properties
    ax.grid(linestyle='--')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(interval)

    # Rotate the tick labels at an angle
    plt.xticks(rotation=45)

    # Use a smaller font size for the tick labels
    plt.xticks(fontsize=8)
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')

    # Plot the true values and the predictions
    ax.plot(X_test_date_sliced, y_test_sliced, label='True values', color="red")
    ax.plot(X_test_date_sliced, predictions, label='Predictions' ,color="blue")
    ax.set_title("Bitcoin", color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Scaled USD', color='white')
    ax.legend(facecolor='#0c1c23', edgecolor='white', fontsize=12,labelcolor ="white")

    # Show the plot
    plt.show()

# cryptoPredict()

def my_view(request):
  image_url = "{% static 'AAPL.jpg' %}"
  return render(request, 'welcome.html', {'image_url': image_url})


def my_home(request):
  image_url = "{% static 'pexels-adrien-olichon-2387793.jpg' %}"
  return render(request, 'home.html', {'image_url': image_url})

def my_dashboard(request):
  return render(request, 'dashboard.html')
