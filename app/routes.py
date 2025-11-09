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
import functools as ft, reduce

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
        print(f"request.form: {request.form}")
        print(f"request.files: {request.files}")
        button_value = request.form.get('action')
        join_button = request.form.get('join')
        print(f"join_button : {join_button}")
        # if button_value == "Upload":
        #     print('ada uploud')
        #     # check if the post request has the file part
        #     if 'file' not in request.files:
        #         flash('No file part')
        #         return redirect(request.url)
        #     file = request.files['file']
        #     print('file udh bs')
        #     # If the user does not select a file, the browser submits an
        #     # empty file without a filename.
        #     if file.filename == '':
        #         flash('No selected file')
        #         return redirect(request.url)
        #     if file and allowed_file(file.filename):
        #         filename = secure_filename(file.filename)
        #         print(app.config['UPLOAD_FOLDER'])
        #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #         return redirect(url_for('prediction', file=filename))
        #         # prediction(file=filename)
        
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


        print('gurt1')
        # print(f'df : {dataframes}')
        
        if 'upload' in request.files: 
            file = request.files['upload']
            if 'table_1' in dataframes:
                print(f"Length of table_1: {len(dataframes['table_1'])}")

            num_table = request.form.get('table_num')
            print(f"[INFO] Uploaded table value: {num_table}")
            
            if 'upload' not in request.files:
                flash('No file part')
                return redirect(request.url)
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                df = pd.read_csv(file)
                table_index = f'table_{num_table}'
                if table_index not in dataframes:
                    dataframes[table_index] = [df]
                else:
                    dataframes[table_index].append(df)
                for i, df in enumerate(dataframes, 1):
                    print(f"\nTable {i}:\n", df)

        # if join_button == "Join":
        print(f"len table 1 dataframes: {len(dataframes['table_1'])}")
        print(f"request.form: {request.form}")
        if 'submit_join' in request.form:
            # Get which table this form was for
            table_num = request.form.get('table_num')

            join_col_data = {}

            for key, value in request.form.items():
                if key.startswith('join_col_'):
                    index = int(key.split('_')[-1])
                    join_col_data[index] = value

            sorted_join_cols = [join_col_data[k] for k in sorted(join_col_data.keys())]

            print(f"User wants to join Table {table_num}")
            print(f"Selected columns in order: {sorted_join_cols}")

            standard_name = sorted_join_cols[0]  # or choose whichever name you want to use consistently

            for i, df in enumerate(dataframes[f'table_{table_num}']):
                rename_dict = {col: standard_name for col in df.columns if col in sorted_join_cols}
                dataframes[f'table_{table_num}'][i] = df.rename(columns=rename_dict)

            # Then merge them
            df_final = ft.reduce(lambda left, right: pd.merge(left, right, on=standard_name), dataframes[f'table_{table_num}'])

            dataframes[f'table_{table_num}'] = [df_final]
            print(df_final.head())

        if 'upload_table' in request.files:
            print('after upload')
            file = request.files['upload_table']
            print('after upload')
            num_table = len(dataframes) + 1
            print(f"[INFO] Uploaded table value: {num_table}")
            
            if 'upload_table' not in request.files:
                flash('No file part')
                return redirect(request.url)
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                df = pd.read_csv(file)
                table_index = f'table_{num_table}'
                if table_index not in dataframes:
                    dataframes[table_index] = [df]
                else:
                    dataframes[table_index].append(df)
                for i, df in enumerate(dataframes, 1):
                    print(f"\nTable {i}:\n", df)
        
        if 'action' in request.form:
            column_mapping = {
                'Amount': 'amount',
                'transactionAmount': 'amount',
                'amt': 'amount',
                'Location': 'location',
                'StateName': 'location',
                'state': 'location',
                'IsFraud': 'is_fraud',
                'Fraud': 'is_fraud',
                'Fraudulent': 'is_fraud'
            }
            # Rename columns in each dataframe inside the dictionary
            for table_name, df_list in dataframes.items():
                for i, df in enumerate(df_list):
                    df_list[i] = df.rename(columns=column_mapping)
            
            # Flatten all dataframes into a single list
            all_dfs = [df for dfs in dataframes.values() for df in dfs]

            df_concat = pd.concat(all_dfs, ignore_index=True)
            print(f'hasil omegad dan len concat: {len(df_concat)}')
            # Print length of each dataframe
            for table_name, df_list in dataframes.items():
                for i, df in enumerate(df_list):
                    print(f"{table_name}[{i}] length: {len(df)}")
            print(df_concat.head())

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