import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application = Flask(__name__)
app = application

## import ridge regression model nd standard scaler
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))    

#Creating route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
       if request.method=='POST':
           #Names must be same as in home.html
           Temperature = request.form.get('Temperature')
           RH = request.form.get('RH')
           Ws = request.form.get('Ws')
           Rain = request.form.get('Rain')
           FFMC = request.form.get('FFMC')
           DMC = request.form.get('DMC')
           ISI = request.form.get('ISI')
           Classes = request.form.get('Classes')
           Region = request.form.get('Region')  
        
           #Standardizing the input data
           new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
           result = ridge_model.predict(new_data_scaled)
           
           return render_template('home.html', results = result[0])     #since result is a list
       #GET method
       else:
        print("GET request to /predictdata received, rendering home.html")
        return render_template('home.html') 
       
        
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)