from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

application=app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])    
def predict():
    if request.method=='POST':
        try:
            
    else:
        return render_template('home.html')