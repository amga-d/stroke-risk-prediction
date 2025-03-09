import joblib
import pandas as pd

# Load the trained model, encoders, and scaler
model = joblib.load("stroke_risk_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Example user input
user_input = {
    "gender": "Female",
    "age": 12,
    "hypertension": "Yes",
    "heart_disease": "No",
    "ever_married": "Yes",
    "work_type": "Self-employed",
    "residence_type": "Urban",
    "avg_glucose_level": 12,
    "bmi": 12,
    "smoking_status": "formerly smoked"
}


input_df = pd.DataFrame([user_input])
for col in encoders.keys():
    input_df[col] = encoders[col].transform(input_df[col]);

# Scale input features
input_scaled = scaler.transform(input_df)

# Predict stroke risk
# prediction = model.predict(input_scaled)
# print("Stroke Risk Prediction:", "High" if prediction[0] == 1 else "Low")

# Get probability of stroke (class 1)
stroke_probability = model.predict_proba(input_scaled)[0][1]

# Print result as a percentage
print(f"\nStroke Risk Probability: {stroke_probability:.2%}")
