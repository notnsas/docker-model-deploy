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
import os
import joblib
import pandas as pd
import numpy as np
import functools as ft
from pydantic import ValidationError
from pandantic import Pandantic
from .utils import Preprocess, Fraud
import pandas as pd
import numpy as np

dataframes = {}

options = [
    "San Antonio",
    "Dallas",
    "New York",
    "Philadelphia",
    "Phoenix",
    "Chicago",
    "San Jose",
    "San Diego",
    "Houston",
    "Los Angeles",
    "California",
    "New Mexico",
    "Puerto Rico",
    "Tennessee",
    "Oregon",
    "Arkansas",
    "Wisconsin",
    "Minnesota",
    "District of Columbia",
    "North Dakota",
    "Illinois",
    "South Dakota",
    "New Hampshire",
    "Wyoming",
    "Nebraska",
    "Guam",
    "Utah",
    "Connecticut",
    "Delaware",
    "Louisiana",
    "Kentucky",
    "South Carolina",
    "American Samoa",
    "Indiana",
    "North Carolina",
    "Montana",
    "Mississippi",
    "Maine",
    "Arizona",
    "Kansas",
    "Missouri",
    "Nevada",
    "Ohio",
    "Pennsylvania",
    "Georgia",
    "Hawaii",
    "Vermont",
    "Virgin Islands, U.S.",
    "Idaho",
    "Florida",
    "Iowa",
    "Colorado",
    "Washington",
    "West Virginia",
    "Alabama",
    "Oklahoma",
    "Massachusetts",
    "Northern Mariana Islands",
    "Virginia",
    "Michigan",
    "Texas",
    "New Jersey",
    "Maryland",
    "Rhode Island",
    "Alaska",
]

pipeline = joblib.load("ml_model/logistic_regression_inference.joblib")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def inference(data):
    # Ngecek array atau data biasa buat inference
    if isinstance(data, np.ndarray):
        columns = ["amount", "location"]

        X = pd.DataFrame(data, columns=columns)
        y = pipeline.predict(X)
        return y[0]
    else:
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
    # Dapetin datanya
    try:
        data = request.json
        print(f"data : {data}")
        amount = float(data.get("amount"))
        location = data.get("location")
    except Exception as e:
        flash(f"Error : {e}")

    # Ngecek valid apa ga pake pydantics
    try:
        fraud_form = Fraud(
            amount=amount,  # Empty string
            location=location,  # Invalid email
        )
    except ValidationError as e:
        flash(e)

    # Running modelnya
    features = np.array([[amount, location]])
    print("feat")
    pred_class_num = inference(features)

    # Map resultnya
    species_map = {0: "Not Fraud", 1: "Fraud"}
    pred_class = species_map.get(pred_class_num, "Unknown")

    print(f"Prediction: {pred_class}")  # This prints to your CMD

    # Berikan json ke javascript
    return jsonify({"prediction": pred_class})


@app.route("/uploads/<name>")
def download_file(name):
    print(f"name : {name}")
    print(f"name : {name}")
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/prediction/<file>")
def prediction(file):
    print("mulai")
    print(f"file : {file}")
    # Dapetin file path
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file)

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".csv"]:
        df = pd.read_csv(file_path)
    elif file_ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    n = len(df)
    print(f"len :{len(df)}")
    df = inference(df)
    print(f"len after inf :{len(df)}")

    # Kalau terlalu besar rownya excel ga mampu
    if n < 1000000:
        path = "data/fraud_with_prediction.xlsx"
        df.to_excel(path, index=False)
    else:
        path = "data/fraud_with_prediction.csv"
        df.to_csv(path, index=False)

    filename = os.path.basename(path)
    return redirect(url_for("download_file", name=filename))


@app.route("/", methods=["GET", "POST"])
def upload_file():
    pred_class = "None"

    # Upload dataframe ke suatu table
    if request.method == "POST":
        if "upload" in request.files:
            try:
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
            except Exception as e:
                print(f"Error : {e}")

        print(f"len table 1 dataframes: {len(dataframes['table_1'])}")
        print(f"request.form: {request.form}")
        # Join data di suatu table
        if "submit_join" in request.form:
            # Dapetin table mana yang mau dijoin
            table_num = request.form.get("table_num")

            join_col_data = {}

            for key, value in request.form.items():
                if key.startswith("join_col_"):
                    index = int(key.split("_")[-1])
                    join_col_data[index] = value

            sorted_join_cols = [join_col_data[k] for k in sorted(join_col_data.keys())]

            print(f"User wants to join Table {table_num}")
            print(f"Selected columns in order: {sorted_join_cols}")

            standard_name = sorted_join_cols[0]

            for i, df in enumerate(dataframes[f"table_{table_num}"]):
                rename_dict = {
                    col: standard_name for col in df.columns if col in sorted_join_cols
                }
                dataframes[f"table_{table_num}"][i] = df.rename(columns=rename_dict)

            # Merge menggunakan ft
            df_final = ft.reduce(
                lambda left, right: pd.merge(left, right, on=standard_name),
                dataframes[f"table_{table_num}"],
            )

            dataframes[f"table_{table_num}"] = [df_final]
            print(df_final.head())

        # Upload table
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

        # Delete table
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

        # Upload semua dataset di table dan inference
        if "upload_dataset" in request.form:
            # try:
            preprocess = Preprocess(dataframes)
            df_concat = preprocess.preprocessing()
            print("concat")
            # Ngecek valid apa ga
            validator = Pandantic(schema=Fraud)
            print("valid")
            if isinstance(df_concat, str):
                flash(df_concat)
                return redirect(request.url)

            # Pengecekan
            df_concat = validator.validate(dataframe=df_concat, errors="skip")
            print("Check valid print ga")
            # Ngecek kalau ga valid bakal print apa ga validnya

            print(df_concat)
            # Save df_concat temporarily
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_input.csv")
            df_concat.to_csv(temp_path, index=False)

            # Redirect and pass the filename
            return redirect(url_for("prediction", file="temp_input.csv"))
            # except Exception as e:
            #     print(f"Error: {e}")
    return render_template(
        "uploud.html", pred_class=pred_class, dataframes=dataframes, options=options
    )


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
