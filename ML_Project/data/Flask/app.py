from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd

app = Flask(__name__, template_folder='templates')

# ✅ Load trained model and transformer
model = pickle.load(open("model.pkl", "rb"))
ct = joblib.load("feature_values.pkl")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/pred')  # Proceed to form
def pred():
    return render_template("index.html")

@app.route('/out', methods=["POST"])
def output():
    try:
        # ✅ Get and validate age
        age_value = request.form["age"]
        try:
            age = float(age_value)
        except ValueError:
            return f"❌ Error: Invalid age value '{age_value}'"

        # ✅ Extract form inputs (without 'leave')
        data = [[
            age,
            request.form["gender"],
            request.form["self_employed"],
            request.form["family_history"],
            request.form["work_interfere"],
            request.form["no_employees"],
            request.form["remote_work"],
            request.form["tech_company"],
            request.form["benefits"],
            request.form["care_options"],
            request.form["wellness_program"],
            request.form["seek_help"],
            request.form["anonymity"],
            request.form["mental_health_consequence"],
            request.form["phys_health_consequence"],
            request.form["coworkers"],
            request.form["supervisor"],
            request.form["mental_health_interview"],
            request.form["phys_health_interview"],
            request.form["mental_vs_physical"],
            request.form["obs_consequence"]
        ]]

        # ✅ Column names — ensure matches what model expects
        feature_cols = [
            'Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
            'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
            'wellness_program', 'seek_help', 'anonymity',
            'mental_health_consequence', 'phys_health_consequence', 'coworkers',
            'supervisor', 'mental_health_interview', 'phys_health_interview',
            'mental_vs_physical', 'obs_consequence'
        ]

        # ✅ Convert to DataFrame
        df = pd.DataFrame(data, columns=feature_cols)

        # ✅ Transform and predict
        transformed = ct.transform(df)
        prediction = model.predict(transformed)[0]

        result = "requires" if prediction == 1 else "doesn't require"
        return render_template("output.html", y=f"This person {result} mental health treatment.")

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
