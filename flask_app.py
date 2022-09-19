import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

app = Flask(__name__, template_folder="templateFiles", static_folder="staticFiles")

attributes_dict = {
    "stroke_form": [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ],
    "bmi_form": [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "smoking_status",
    ],
    "glucose_form": [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "bmi",
        "smoking_status",
    ],
    "hypertension_form": [
        "gender",
        "age",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ],
    "bg_form": [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "smoking_status",
    ],
    "bh_form": [
        "gender",
        "age",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "smoking_status",
    ],
    "gh_form": [
        "gender",
        "age",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "bmi",
        "smoking_status",
    ],
    "bgh_form": [
        "gender",
        "age",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "smoking_status",
    ],
}
form_models_dict = {
    "stroke_form": ["best_stroke_model_trained", "None", "None"],
    "bmi_form": ["best_optuna_bmi_trained", "None", "None"],
    "glucose_form": ["best_optuna_glucose_trained", "None", "None"],
    "hypertension_form": ["best_optuna_hypertension_trained", "None", "None"],
    "bg_form": ["best_gb_bmi_trained", "best_gb_glucose_trained", "None"],
    "bh_form": ["best_hb_bmi_trained", "best_hb_hypertension_trained", "None"],
    "gh_form": ["best_hg_glucose_trained", "best_hg_hypertension_trained", "None"],
    "bgh_form": [
        "best_hgb_bmi_trained",
        "best_hgb_glucose_trained",
        "best_hgb_hypertension_trained",
    ],
}
models_list = [
    "best_stroke_model_trained",
    "best_optuna_bmi_trained",
    "best_optuna_glucose_trained",
    "best_optuna_hypertension_trained",
    "best_gb_bmi_trained",
    "best_gb_glucose_trained",
    "best_hb_bmi_trained",
    "best_hb_hypertension_trained",
    "best_hg_glucose_trained",
    "best_hg_hypertension_trained",
    "best_hgb_bmi_trained",
    "best_hgb_glucose_trained",
    "best_hgb_hypertension_trained",
]
models_dict = {}
last_form = "stroke_form"
numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]


def load_model():
    global models_dict
    for model in models_list:
        models_dict[model] = joblib.load(f"models\\trained_models\\{model}.pkl")
    models_dict["None"] = "None"


def string_to_numeral(df: pd.DataFrame):
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = int(df[feature])
    return df


@app.route("/")
def index():
    return render_template("index.html", last_form=last_form)


@app.route("/predict", methods=["GET", "POST"])
def get_prediction():
    if request.method == "POST":
        last_form = request.form.get("form_type")
        attributes_list = []
        print(1)
        for elem in attributes_dict[last_form]:
            attributes_list.append(request.form.get(elem))
        df = pd.DataFrame(columns=attributes_dict[last_form])
        df.loc[1] = attributes_list
        df.index.name = "id"
        df = string_to_numeral(df)
        prediction = []
        for model_name in form_models_dict[last_form]:
            print(form_models_dict[last_form])
            if model_name == "None":
                prediction.append("None")
            else:
                print(model_name)
                prediction.append(int(models_dict[model_name].predict(df)))
        print(prediction)
        return redirect(
            url_for(
                "index",
                _anchor="anchor",
                last_form=last_form,
                prediction_1=prediction[0],
                prediction_2=prediction[1],
                prediction_3=prediction[2],
            )
        )
    return render_template("index.html")


if __name__ == "__main__":
    load_model()
    app.debug = True
    app.run(host="0.0.0.0", port=80)
