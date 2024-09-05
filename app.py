from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

with open('fuel.pkl','rb') as f:
    le_fuel=pickle.load(f)
with open('seller.pkl','rb') as f:
    le_seller=pickle.load(f)
with open('tran.pkl','rb') as f:
    le_trans=pickle.load(f)

with open('scalar.pkl','rb') as f:
    scalar=pickle.load(f)
model=pickle.load(open('car_model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')



@app.route('/home1', methods=['POST'])
def home1():
    try:
        Year = float(request.form['a'])
        Present_Price = float(request.form['b'])
        Kms_Driven = float(request.form['c'])
        Fuel_Type = request.form['d']
        Seller_Type = request.form['e']
        Transmission = request.form['f']
        Owner = float(request.form['g'])
        

        fuel_en=le_fuel.transform([Fuel_Type])[0]
        seller_en=le_seller.transform([Seller_Type])[0]
        trans_en=le_trans.transform([Transmission])[0]
        

        input_data = np.array([[Year,Present_Price,Kms_Driven,fuel_en,seller_en,trans_en,Owner]])
        input_data_scaled = scalar.transform(input_data)

        prediction = model.predict(input_data_scaled)[0]

        return render_template('after.html', prediction=prediction)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        # return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)