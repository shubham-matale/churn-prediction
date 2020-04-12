from flask import Flask, redirect, render_template, request, session, abort, Markup
import os
import pandas
import numpy as np
from keras.models import load_model
from sklearn.externals.joblib import load

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)