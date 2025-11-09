from flask import (
    render_template,
    flash,
    redirect,
    url_for,
    send_from_directory,
    request,
    jsonify,
    send_file,
)
from app import app, ALLOWED_EXTENSIONS, db
from app.forms import LoginForm
from werkzeug.utils import secure_filename
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier  # you must import this class
import pandas as pd
import numpy as np
import functools as ft
from functools import reduce
from datetime import date
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel
from pandantic import Pandantic
from .utils import Preprocess, Fraud
import pandas as pd
import numpy as np
import imblearn

# from imblearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
# from joblib import dump
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.ensemble import BalancedBaggingClassifier
# from sklearn.tree import DecisionTreeClassifier

dataframes = {}

pipeline = joblib.load("ml_model/logistic_regression_inference.joblib")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def inference(data):
    # model = joblib.load("ml_model/random_forest_iris.pkl")

    if isinstance(data, np.ndarray):
        columns = ["amount", "location"]

        X = pd.DataFrame(data, columns=columns)
        y = pipeline.predict(X)
        return y[0]
    else:
        # df = pd.read_csv(data)
        print("before drop")
        if "is_fraud" in data.columns.tolist():
            X = data.drop("is_fraud", axis=1)
        else:
            X = data
        print("after")
        data["is_fraud_prediction"] = pipeline.predict(X)
        return data


@app.route("/api/prediction", methods=["POST"])
def api_prediction():
    print("test")
    data = request.json
    print(f"data : {data}")
    amount = float(data.get("amount"))
    location = data.get("location")

    # 3. Run the model
    features = np.array([[amount, location]])
    print("feat")
    pred_class_num = inference(features)

    # 4. Map the result
    species_map = {0: "Not Fraud", 1: "Fraud"}
    pred_class = species_map.get(pred_class_num, "Unknown")

    print(f"Prediction: {pred_class}")  # This prints to your CMD

    # 5. THE FIX: Send JSON back to the JavaScript
    return jsonify({"prediction": pred_class})


@app.route("/uploads/<name>")
def download_file(name):
    print(f"name : {name}")
    print(f"name : {name}")
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)
    # return send_from_directory(
    #     directory=r"D:\Code\Kuliah\Teknik Pengembangan model\Docker Model Deployment\data",
    #     path=name,
    #     as_attachment=True,
    # )


@app.route("/prediction/<file>")
def prediction(file):
    print("mulai")
    # file = request.args.get("file")
    # file_path = os.path.join(app.config["UPLOAD_FOLDER"], file)
    # file = request.args.get("file")
    print(f"file : {file}")
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file)

    df = pd.read_excel(file_path)
    df = inference(df)

    df.to_excel("data/fraud_with_prediction.xlsx", index=False)
    filename = secure_filename("data/fraud_with_prediction.csv")
    print(f"app config :{app.config["UPLOAD_FOLDER"]}")
    # download_file(name="fraud_with_prediction.xlsx")
    # redirect(url_for("download_file", name=filename))
    return redirect(url_for("download_file", name="fraud_with_prediction.xlsx"))


@app.route("/", methods=["GET", "POST"])
def upload_file():
    pred_class = "None"
    if request.method == "POST":
        print(f"request.form: {request.form}")
        print(f"request.files: {request.files}")
        button_value = request.form.get("action")
        join_button = request.form.get("join")
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
        print(f"button_value :{button_value}")
        if button_value == "submit":
            data = request.json
            print(f"data: {data}")

            amount = float(data.get("amount"))
            location = str(data.get("location"))
            # petal_length = float(data.get("petal_length"))
            # petal_width = float(data.get("petal_width"))

            print(f"amount : {amount}")
            # model = joblib.load('ml_model/random_forest_iris.pkl')
            features = np.array([[amount, location]])
            # pred_class_num = model.predict(features)
            pred_class_num = inference(features)
            species_map = {0: "Not Fraud", 1: "Fraud"}
            pred_class = species_map.get(pred_class_num, "Unknown")

        print("gurt1")
        # print(f'df : {dataframes}')

        if "upload" in request.files:
            file = request.files["upload"]
            if "table_1" in dataframes:
                print(f"Length of table_1: {len(dataframes['table_1'])}")

            num_table = request.form.get("table_num")
            print(f"[INFO] Uploaded table value: {num_table}")

            if "upload" not in request.files:
                flash("No file part")
                return redirect(request.url)
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                df = pd.read_csv(file)
                table_index = f"table_{num_table}"
                if table_index not in dataframes:
                    dataframes[table_index] = [df]
                else:
                    dataframes[table_index].append(df)
                for i, df in enumerate(dataframes, 1):
                    print(f"\nTable {i}:\n", df)

        # if join_button == "Join":
        print(f"len table 1 dataframes: {len(dataframes['table_1'])}")
        print(f"request.form: {request.form}")
        if "submit_join" in request.form:
            # Get which table this form was for
            table_num = request.form.get("table_num")

            join_col_data = {}

            for key, value in request.form.items():
                if key.startswith("join_col_"):
                    index = int(key.split("_")[-1])
                    join_col_data[index] = value

            sorted_join_cols = [join_col_data[k] for k in sorted(join_col_data.keys())]

            print(f"User wants to join Table {table_num}")
            print(f"Selected columns in order: {sorted_join_cols}")

            standard_name = sorted_join_cols[
                0
            ]  # or choose whichever name you want to use consistently

            for i, df in enumerate(dataframes[f"table_{table_num}"]):
                rename_dict = {
                    col: standard_name for col in df.columns if col in sorted_join_cols
                }
                dataframes[f"table_{table_num}"][i] = df.rename(columns=rename_dict)

            # Then merge them
            df_final = ft.reduce(
                lambda left, right: pd.merge(left, right, on=standard_name),
                dataframes[f"table_{table_num}"],
            )

            dataframes[f"table_{table_num}"] = [df_final]
            print(df_final.head())

        if "upload_table" in request.files:
            print("after upload")
            file = request.files["upload_table"]
            print("after upload")
            num_table = len(dataframes) + 1
            print(f"[INFO] Uploaded table value: {num_table}")

            if "upload_table" not in request.files:
                flash("No file part")
                return redirect(request.url)
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                df = pd.read_csv(file)
                table_index = f"table_{num_table}"
                if table_index not in dataframes:
                    dataframes[table_index] = [df]
                else:
                    dataframes[table_index].append(df)
                for i, df in enumerate(dataframes, 1):
                    print(f"\nTable {i}:\n", df)

        if "delete" in request.form:
            table_num = int(request.form.get("table_num"))
            del dataframes[f"table_{table_num}"]

            # Reindex subsequent tables
            max_index = len(dataframes) + 1  # +1 because we just deleted one
            for i in range(table_num + 1, max_index + 1):
                old_key = f"table_{i}"
                new_key = f"table_{i - 1}"
                if old_key in dataframes:
                    dataframes[new_key] = dataframes.pop(old_key)
        if "action" in request.form:
            preprocess = Preprocess(dataframes)
            df_concat = preprocess.preprocessing()
            print("concat")
            # Ngecek valid apa ga
            validator = Pandantic(schema=Fraud)
            print("valid")
            # Pengecekan
            df_concat = validator.validate(dataframe=df_concat, errors="skip")

            print(df_concat)
            # Save df_concat temporarily
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_input.xlsx")
            df_concat.to_excel(temp_path, index=False)

            # Redirect and pass the filename
            return redirect(url_for("prediction", file="temp_input.xlsx"))
    return render_template("uploud.html", pred_class=pred_class, dataframes=dataframes)


@app.route("/index")
def index():
    user = {"username": "Nara"}
    posts = [
        {"author": {"username": "John"}, "body": "Beatiful day in Portland or sum!"},
        {"author": {"username": "Susan"}, "body": "The avenger so cool sum!"},
    ]
    return render_template("index.html", title="Home", user=user, posts=posts)


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash(
            "Login requested for user {}, remember_me={}".format(
                form.username.data, form.remember_me.data
            )
        )
        return redirect(url_for("index"))
    return render_template("login.html", title="Sign in", form=form)
