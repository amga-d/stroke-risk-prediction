import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import joblib

# Load dateset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

#  Change all the columns names into lowercase
df.columns = df.columns.str.lower()

# Drop irrelevent columns (ID)
df.drop(columns=['id'],inplace =True)

# Define categorical columns
categorical_columns = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
binary_columns = ["hypertension", "heart_disease"] # Binary categorical columns

# Store Encoders for later use
encoders ={}

# Encode main categorical columns
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col]) # Fit & transform

binary_categories = ["No","Yes"]

# Encode Binary categorical coulmns ("hypertension", "heart_disease")
for col in binary_columns:
    encoders[col] = LabelEncoder()
    encoders[col].fit(binary_categories) # Fit only (values are Tranformed)
    categorical_columns.append(col) # Add to categorical columns

category_mappings = {
    col: {label: idx for idx, label in enumerate(encoders[col].classes_)}
    for col in categorical_columns
}

print(category_mappings)


print("\nCategory Mappings:\n")
for col, mapping in category_mappings.items():
    print(f"{col}:")
    for label, idx in mapping.items():
        print(f"  {label} -> {idx}")
    print("-" * 30)  # Add a separator for better readability

# Handle missing values in BMI
df["bmi"] = df['bmi'].fillna(df["bmi"].median())


# Seprate features and target
x = df.drop(columns=["stroke"])
y = df['stroke']

# Split data into training and testing
X_train ,X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)


# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train);

y_pred = model.predict(X_test)


print("Accuracy", accuracy_score(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))

# Save the model
joblib.dump(model, "stroke_risk_model.pkl")

# Save the encoder and scalar
joblib.dump(scaler, "scalar.pkl")
joblib.dump(encoders,"encoders.pkl")

print("Model and preprocessing tools saved successfully!")

