# Stroke Risk Prediction Model

## Project Overview

This machine learning project uses a Random Forest classifier to predict stroke risk based on health and demographic indicators. The model analyzes features such as age, gender, health conditions, and lifestyle factors to determine an individual's stroke risk.

## Dataset

The model is trained on the "healthcare-dataset-stroke-data.csv" dataset which includes:

- **gender**: Sex of the patient
- **age**: Age of the patient
- **hypertension**: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- **heart_disease**: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- **ever_married**: Marital status
- **work_type**: Type of occupation
- **residence_type**: Urban or Rural residence
- **avg_glucose_level**: Average glucose level in blood
- **bmi**: Body Mass Index
- **smoking_status**: Smoking status of the patient
- **stroke**: Target variable (1 if the patient had a stroke, 0 if not)

## Installation

```bash
# Clone the repository
git clone https://github.com/amga-d/stroke-risk-prediction.git
cd stroke-risk-prediction

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- pandas
- numpy
- scikit-learn
- joblib

## Usage

### Training the Model

To train the model, run:

```bash
python model.py
```

This script:

1. Loads and preprocesses the dataset
2. Encodes categorical features
3. Handles missing values
4. Trains a Random Forest classifier
5. Saves the model and preprocessing tools

### Making Predictions

You can create a script to make predictions with new patient data:

```python
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load("stroke_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# Example patient data
patient = {
    "gender": "Female",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 92.0,
    "bmi": 24.3,
    "smoking_status": "never smoked"
}

# Process the input data (similar to model.py preprocessing)
# ...

# Make prediction
risk_prediction = model.predict(processed_input)[0]
risk_level = "High" if risk_prediction == 1 else "Low"
print(f"Stroke Risk: {risk_level}")
```

## Model Performance

The Random Forest model achieves the following performance metrics on the test set:

```
Accuracy: [Your model's accuracy]
Classification Report:
            precision     recall  f1-score   support
        0       0.95       0.99      0.97      972
        1       0.17       0.02      0.04      50
```

## Feature Encoding

The model uses LabelEncoder to transform categorical variables with the following mappings:

### Gender

- Female → 0
- Male → 1
- Other → 2

### Ever Married

- No → 0
- Yes → 1

### Work Type

- Govt_job → 0
- Never_worked → 1
- Private → 2
- Self-employed → 3
- children → 4

### Residence Type

- Rural → 0
- Urban → 1

### Smoking Status

- Unknown → 0
- formerly smoked → 1
- never smoked → 2
- smokes → 3

### Hypertension

- No → 0
- Yes → 1

---

### Heart Disease:

- No → 0
- Yes → 1

## Files

- model.py: Main script for training and saving the model
- healthcare-dataset-stroke-data.csv: Dataset file
- stroke_risk_model.pkl: Trained Random Forest model
- scaler.pkl: StandardScaler for numerical features
- encoders.pkl: LabelEncoders for categorical features

## Future Improvements

- Implement hyperparameter tuning
- Build a web interface for predictions
