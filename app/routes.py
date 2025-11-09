from flask import render_template, flash, redirect, url_for, send_from_directory, request, jsonify
from app import app, ALLOWED_EXTENSIONS, db
from app.forms import LoginForm
from app.models import Fraud
from werkzeug.utils import secure_filename
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier  # you must import this class
import pandas as pd
import numpy as np


dataframes = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def inference(data):
    model = joblib.load('ml_model/random_forest_iris.pkl')

    if isinstance(data, np.ndarray):
        X = data
        return model.predict(X)[0]
    else:
        df = pd.read_csv(data)
        
        X = df.drop("target", axis=1).values  
        y = df["target"]

        df["predicted"] = model.predict(X)
        return df
    return None

@app.route('/api/prediction', methods=['POST'])
def api_prediction():
    print('test')
    data = request.json
    sepal_length = float(data.get('sepal_length'))
    sepal_width = float(data.get('sepal_width'))
    petal_length = float(data.get('petal_length'))
    petal_width = float(data.get('petal_width'))

    # 3. Run the model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class_num = inference(features)
    
    # 4. Map the result
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    pred_class = species_map.get(pred_class_num, "Unknown")
    
    print(f"Prediction: {pred_class}") # This prints to your CMD

    # 5. THE FIX: Send JSON back to the JavaScript
    return jsonify({
        'prediction': pred_class
    })

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/prediction')
def prediction():
    file = request.args.get('file')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    df = inference(file_path)

    df.to_excel("data/iris_with_predictions.xlsx", index=False)
    filename = secure_filename("data/iris_with_predictions.csv")
    download_file(name="iris_with_predictions.xlsx")

    return redirect(url_for('download_file', name="iris_with_predictions.xlsx"))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_class = "None"
    if request.method == 'POST':
        button_value = request.form.get('action')
        join_button = request.form.get('join')
        print(f"join_button : {join_button}")
        if button_value == "Upload":
            print('ada uploud')
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            print('file udh bs')
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print(app.config['UPLOAD_FOLDER'])
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('prediction', file=filename))
                # prediction(file=filename)
        
        if button_value == "submit":
            data = request.json

            sepal_length = float(data.get('sepal_length'))
            sepal_width = float(data.get('sepal_width'))
            petal_length = float(data.get('petal_length'))
            petal_width = float(data.get('petal_width'))

            print(f"sepal_length : {sepal_length}")
            # model = joblib.load('ml_model/random_forest_iris.pkl')
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            # pred_class_num = model.predict(features)
            pred_class_num = inference(features)
            species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
            pred_class = species_map.get(pred_class_num, "Unknown")


        # df1 = pd.DataFrame({
        #     "location": ["Alice", "Bob", "Charlie"],
        #     "amount": [10, 10.1, 2.5],
        #     "fraud": [1, 0, 1],
        #     "email": ["a@x.com", "b@x.com", "c@x.com"]
        # })

        # # 2️⃣ Second table
        # df2 = pd.DataFrame({
        #     "email": ["a@x.com", "b@x.com", "c@x.com"],
        #     "amt": [100, 200, 150]
        # })

        # # Add both to list
        # dataframes.append([df1, df2])
        print('gurt1')
        print(f'df : {dataframes}')
        file = request.files['upload']
        if file:
            num_table = request.form.get('upload')
            print(f'gurt : {file}')
            
            if 'upload' not in request.files:
                flash('No file part')
                return redirect(request.url)
            # file = request.files['upload']
            print('file udh bs')
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                df = pd.read_csv(file)
                table_index = f'table_{num_table}'
                if table_index not in dataframes:
                    dataframes[table_index] = [df]
                else:
                    dataframes[table_index].extend(df)
                # print(f"df : P{dataframes}")
                # ✅ Verify
                for i, df in enumerate(dataframes, 1):
                    print(f"\nTable {i}:\n", df)
        # if join_button == "Join":
            


    return render_template('uploud.html', pred_class=pred_class, dataframes=dataframes)


@app.route('/index')
def index():
    user = {'username': 'Nara'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beatiful day in Portland or sum!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The avenger so cool sum!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash("Login requested for user {}, remember_me={}".format(
            form.username.data, form.remember_me.data
        ))
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign in', form=form)