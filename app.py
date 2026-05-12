from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')

MODEL_PATH = os.environ.get('MODEL_PATH', './models/random_forest_kaggle.pkl')
_uploaded_path = '/mnt/data/random_forest_kaggle.pkl'
if not os.path.exists(MODEL_PATH) and os.path.exists(_uploaded_path):
    MODEL_PATH = _uploaded_path

# load safely
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded OK from:", MODEL_PATH)
except Exception as e:
    model = None
    print("Model load failed:", e)

# default form values
DEFAULT_FORM = {
    "Area": "India",
    "Year": "2004",
    "average_rain_fall_mm_per_year": "1500",
    "pesticides_tonnes": "180",
    "avg_temp": "29"
}

@app.route('/')
def index():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Render the form (GET)
    if request.method == 'GET':
        return render_template(
            'predict_form.html',
            model_loaded=(model is not None),
            result=None,
            form_values=DEFAULT_FORM
        )

    # Handle POST (JSON or form)
    if request.is_json:
        data = request.get_json()
        from_json = True
    else:
        data = request.form.to_dict()
        from_json = False

    # normalize/validate types for numeric fields
    # we will preserve whatever text input user gave so we can show it back
    form_values = DEFAULT_FORM.copy()
    for k in form_values.keys():
        if k in data and data[k] is not None:
            form_values[k] = str(data[k])

    try:
        # convert numeric fields for prediction (if present)
        if 'Year' in data:
            data['Year'] = int(data['Year'])
        if 'average_rain_fall_mm_per_year' in data:
            data['average_rain_fall_mm_per_year'] = float(data['average_rain_fall_mm_per_year'])
        if 'pesticides_tonnes' in data:
            data['pesticides_tonnes'] = float(data['pesticides_tonnes'])
        if 'avg_temp' in data:
            data['avg_temp'] = float(data['avg_temp'])
    except Exception as e:
        # return error JSON for JSON clients; render form with error for form clients
        if from_json:
            return jsonify({'error': f'Bad input types: {e}'}), 400
        return render_template(
            'predict_form.html',
            model_loaded=(model is not None),
            result=f"Bad input types: {e}",
            form_values=form_values
        )

    # model presence check
    if model is None:
        if from_json:
            return jsonify({'error': 'Model not loaded on server.'}), 500
        return render_template(
            'predict_form.html',
            model_loaded=False,
            result="Model not loaded on server.",
            form_values=form_values
        )

    # Build DataFrame and predict
    try:
        df = pd.DataFrame([data])
        pred = model.predict(df)
        prediction_value = float(pred[0])
    except Exception as e:
        if from_json:
            return jsonify({'error': str(e)}), 400
        return render_template(
            'predict_form.html',
            model_loaded=True,
            result=f"Prediction error: {e}",
            form_values=form_values
        )

    # If JSON client requested prediction -> return JSON
    if from_json:
        return jsonify({'prediction': prediction_value})

    # For HTML form: render the same form but show result and keep values
    return render_template(
        'predict_form.html',
        model_loaded=True,
        result=prediction_value,
        form_values=form_values
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
