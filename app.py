from flask import Flask, render_template, request, redirect

#ignore user warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pickle

app = Flask(__name__)
#load saved model
with open('flood_model.pkl' , 'rb') as f1:
    flood_model = pickle.load(f1)
#load saved scaler
with open('scaler.pkl', 'rb') as f2:
    scaler = pickle.load(f2)

@app.route('/')
def home():
    return render_template('home.html', img_url='./static/images/flood_img.jpg')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return redirect('/')
    annual = float(request.form.get('annual'))
    jf = float(request.form.get('jf'))
    mam = float(request.form.get('mam'))
    jjas = float(request.form.get('jjas'))
    ond = float(request.form.get('ond'))
    
    input_data = np.array([annual, jf, mam, jjas, ond]).reshape(1,-1)
    scaled_data = scaler.transform(input_data)
    
    pred = flood_model.predict(scaled_data)
    print(scaled_data)
    print(pred)
    if pred == 0:
        pred_text = "No Chance for Flood"
        return render_template('predict.html', img_url='./static/images/normal_sky.jpg', prediction=pred_text)
    
    elif pred == 1:
        pred_text = "High Chance for Flood"
        return render_template('predict.html', img_url='./static/images/flood_warning.png', prediction=pred_text)
    


    
if __name__ == '__main__':
    app.run()
    
    