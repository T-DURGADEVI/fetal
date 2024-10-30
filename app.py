from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import pickle
import joblib
import webbrowser
app = Flask(__name__)

# Load the saved models and scalers
fetal_bw_m1 = joblib.load('models/best_catboost_model.pkl')
fetal_bw_m2 = joblib.load('models/best_lgbm_model.pkl')
model = load_model('models/fetalcnn.h5')

default_features = {
    "heart": 0.0,
    "acceleration": 0.0,
    "movement": 0.0,
    "contractions": 0.0,
    "light_declare": 0.0,
    "severe_declare": 0.0,
    "prolong_declare": 0.0,
    "short_variability": 0.0,
    "short_variability_mean": 0.0,
    "long_variability": 0.0,
    "long_variability_mean": 0.0,
    "histogram_width": 0.0,
    "histogram_min": 0.0,
    "histogram_max": 0.0,
    "peaks": 0.0,
    "zeroes": 0.0,
    "histogram_mode": 0.0,
    "histogram_mean": 0.0,
    "histogram_median": 0.0,
    "histogram_variance": 0.0,
    "histogram_tendency": 0.0
}



desired_input_shape = (len(default_features),)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/weight1',methods=['POST'])
def weight1():
    if request.method == 'POST':
        gagef = float(request.form['gage'])
        magef = float(request.form['mage'])
        mheightf = float(request.form['mheight'])
        mweightf = float(request.form['mweight'])
        smokef = 1 if request.form['smoke'] == "Yes" else 0
        parityf = 1 if request.form['parity'] == "Yes" else 0

        data = np.array([[gagef, parityf, magef, mheightf, mweightf, smokef]])
        y_pred1 = fetal_bw_m1.predict(data)
        y_pred2 = fetal_bw_m2.predict(data)
        y_pred = [np.round((1 * i + 5 * j) / 6.0, 3) for i, j in zip(y_pred1, y_pred2)]
        final_prediction = np.array(y_pred)[0]
        final_prediction_kg=final_prediction*0.0283

        if final_prediction_kg<2.6:
            message='Under Weight'
        elif final_prediction_kg>=2.6 and final_prediction_kg<3.6:
            message='Normal Weight'
        else:
            message='Over Weight'
        return render_template('fetwe_result.html', prediction=final_prediction_kg, cat_messages=message)

@app.route('/predict', methods=['POST'])
def predict():
    features = {}
    features = {feature_name: float(request.form.get(feature_name, default_features[feature_name])) for feature_name in default_features}

    feature_array = np.array(list(features.values())).reshape(1, -1)

    # Make predictions using the pre-trained model
    predictions = model.predict(feature_array)

    # Map model predictions to 1, 2, or 3 and pass the prediction to the result page
    prediction_class = int(np.argmax(predictions)) + 1  
    if prediction_class==1:
        message='Normal'
    elif prediction_class==2:
        message='Suspect'
    else:
        message='Pathological'
    return render_template('fetal_hc_result.html', prediction=prediction_class,ermessage=message)
       


@app.route('/index.html')
def index1():
    return render_template('index.html')

@app.route('/fetal_we_fetal_hc.html')
def fetwh():
   return render_template( 'fetal_we_fetal_hc.html')


@app.route('/health.html')
def health():
    return render_template('health.html')

@app.route('/weight.html')
def weight():
    return render_template('weight.html')


if __name__ == '__main__':

    webbrowser.open_new('http://127.0.0.1:5000/')
    
    app.run()
