from time import sleep

from flask import Flask, request, render_template
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import load_model
import json
# from werkzeug import secure_filename
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from keras.engine.saving import load_model
import csv
import subprocess

app = Flask(__name__)

dropdown_list = []
dropdown_list_2 = []


def model_prection():
    churn = pd.read_excel("churn.xlsx")
    churn.head()
    churn.info()
    churn.columns[churn.isnull().any()]
    churn.select_dtypes(exclude=[np.number]).head()
    X = churn.iloc[:, 4:20].values
    Y = churn.iloc[:, 20].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
    labelencoder_X_2 = LabelEncoder()
    X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    cor = churn.corr()
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    clf = Sequential()
    clf.add(Dense(units=24, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001),
                  input_dim=16))
    clf.add(Dense(units=24, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
    clf.add(Dense(units=24, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
    clf.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = clf.fit(X, Y, batch_size=20, epochs=250)
    data_to_predict = preprocess_data();
    prediction = clf.predict(data_to_predict)
    return prediction;


def sav_name(n):
    with open("filename.txt", "w") as ff:
        ff.write(n)


def preprocess_data():
    ffr = open("filename.txt", "r")
    upl_file = ffr.read()
    upl_file = str(upl_file)
    churn = pd.read_csv(upl_file)
    churn.head()
    churn.select_dtypes(exclude=[np.number]).head()
    X = churn.iloc[:, 4:20].values
    # Y = churn.iloc[:, 20].values
    labelencoder_X_1 = LabelEncoder()
    X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
    labelencoder_X_2 = LabelEncoder()
    X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
    # labelencoder_Y = LabelEncoder()
    # Y = labelencoder_Y.fit_transform(Y)
    cor = churn.corr()
    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/getDetails', methods=['GET'])
def get_details():
    if request.method == 'GET':
        if 'phone_number' in request.args:
            mobile_number = request.args['phone_number']
        else:
            response = {}
            response['success']='false'
            response['msg']="Id Not Available"

            return
        print(mobile_number)
        ffr = open("filename.txt", "r")
        upl_file = ffr.read()
        upl_file = str(upl_file)
        data = preprocess_data()
        data_list = []
        header_list = []
        with open('testtest1.csv') as file:
            allRead = csv.reader(file, delimiter=',')
            lineCount = 0
            for row in allRead:
                # print(row)
                if lineCount == 0:
                    header_list = row
                    lineCount = lineCount + 1
                else:
                    # print('de',mobile_number,row[3])
                    if mobile_number == row[4]:
                        lineCount = lineCount + 1
                        data_list.append(row)
            response = {}
            response['success'] = 'true'
            response['data']={}
            response['data']['data_list'] = data_list
            response['data']['header_list'] = header_list
        return response


@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        # K.clear_session()
        dropdown_list.clear()
        f = request.files['file']
        # print(f)
        f.save(f.filename)
        sav_name(f.filename)
        ffr = open("filename.txt", "r")
        upl_file = ffr.read()
        upl_file = str(upl_file)
        data = preprocess_data()
        # model = load_model('model.h5')
        # model._make_predict_function()
        # batcmd = "python3 Churn\ analysis.py"
        # y_pred = subprocess.check_output(batcmd, shell=True)
        # print(y_pred)
        # y_pred = model_prection();
        classifier = load_model('my_model.h5')
        prediction = classifier.predict(data)
        percentage = []
        churn = []
        for i in range(0, len(prediction)):
            if prediction[i]>0.5:
                churn.append('True')
                percentage.append(prediction[i][0] * 100)
                prediction[i] = 1
            else:
                churn.append('False')
                percentage.append(prediction[i][0] * 100)
                prediction[i] = 0
        data_list = []
        header_list = []
        dff = pd.read_csv(upl_file)
        dff.drop(["churn"], axis=1, inplace=True)
        dff['Stays Or Left'] = churn
        dff['Percentage'] = percentage
        dff.to_csv('testtest1.csv')
        with open('testtest1.csv') as file:
            allRead = csv.reader(file, delimiter=',')
            lineCount = 0
            for row in allRead:
                print(row)
                if lineCount == 0:
                    header_list = row
                    lineCount = lineCount + 1
                else:
                    lineCount = lineCount + 1
                    data_list.append(row)
        return render_template('tableview.html', data_list=data_list, col_len=len(data_list[0]),
                               header_list=header_list)

if __name__ == '__main__':
    app.run(debug=False)

