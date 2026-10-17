
from flask import Flask, render_template, request
import pandas as pd
import pickle

#app = Flask(__name__)
import os
from flask import Flask, render_template

app = Flask(
    __name__,
    template_folder=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "templates"
    )
)
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])

    data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks]
    })

    prediction = model.predict(data)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
