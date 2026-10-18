
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

with open(model_path, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def index():
    return render_template(
        "index.html",
        prediction=None,
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the HTML form
        hours = float(request.form["Hours_Studied"])
        attendance = float(request.form["Attendance_Percent"])
        previous_score = float(request.form["Previous_Score"])
        assignments = float(request.form["Assignments_Completed"])
        sleep = float(request.form["Sleep_Hours"])
        activities = int(
            request.form["Extracurricular_Activities_up"]
        )

        # Validate inputs
        if not 0 <= hours <= 8:
            raise ValueError("Study hours must be between 0 and 8.")

        if not 50 <= attendance <= 100:
            raise ValueError("Attendance must be between 50 and 100.")

        if not 30 <= previous_score <= 100:
            raise ValueError("Previous score must be between 30 and 100.")

        if not 0 <= assignments <= 10:
            raise ValueError("Assignments must be between 0 and 10.")

        if not 4 <= sleep <= 10:
            raise ValueError("Sleep hours must be between 4 and 10.")

        if activities not in (0, 1):
            raise ValueError("Invalid extracurricular activity selection.")

        # Create input DataFrame in the same column order
        input_data = pd.DataFrame([{
            "Hours_Studied": hours,
            "Attendance_Percent": attendance,
            "Previous_Score": previous_score,
            "Assignments_Completed": assignments,
            "Sleep_Hours": sleep,
            "Extracurricular_Activities_up": activities
        }])

        # Predict the performance score
        prediction = float(model.predict(input_data)[0])

        return render_template(
            "index.html",
            prediction=round(max(0, min(100, prediction)), 2),
            error=None
        )

    except (ValueError, KeyError) as e:
        return render_template(
            "index.html",
            prediction=None,
            error=str(e)
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            prediction=None,
            error="Prediction failed. Check that model.pkl matches the input features."
        )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)