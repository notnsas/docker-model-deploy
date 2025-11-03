from flask import render_template, flash, redirect, url_for, send_from_directory, request
from app import app, ALLOWED_EXTENSIONS
from app.forms import LoginForm
from werkzeug.utils import secure_filename
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier  # you must import this class
import pandas as pd
import numpy as np

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/prediction')
def prediction():
    file = request.args.get('file')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    model = joblib.load('ml_model/random_forest_iris.pkl')

    df = pd.read_csv(file_path)
    
    # 2. Separate features and target
    X = df.drop("target", axis=1).values  
    y = df["target"]

    # 4. Predict on the same data (or any new data)
    predictions = model.predict(X)

    # 5. Add predictions to DataFrame
    df["predicted"] = predictions

    # Optional: save predictions
    df.to_excel("data/iris_with_predictions.xlsx", index=False)
    filename = secure_filename("data/iris_with_predictions.csv")
    return redirect(url_for('download_file', name="iris_with_predictions.xlsx"))

    # return render_template('prediction.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    pred_class = "None"
    if request.method == 'POST':
        button_value = request.form.get('action')
        if button_value == "Upload":
            print('ada uploud')
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
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
        
        if button_value == "submit":
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])
            model = joblib.load('ml_model/random_forest_iris.pkl')
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred_class_num = model.predict(features)[0]

            species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
            pred_class = species_map.get(pred_class_num, "Unknown")

    return render_template('uploud.html', pred_class=pred_class)


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