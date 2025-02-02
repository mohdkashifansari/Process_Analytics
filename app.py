import streamlit as st
import numpy as np
import json
import pickle

# Load model and metadata
model_filename = "sepsis_activity_model.pkl"
columns_filename = "columns.json"

with open(model_filename, "rb") as f:
    rf_model = pickle.load(f)

with open(columns_filename, "r") as f:
    column_data = json.load(f)

# Extract saved metadata
max_seq_length = column_data["max_seq_length"]
activity_encoder = column_data["activity_encoder"]
label_classes = column_data["label_classes"]
pca_feature_names = column_data["pca_features"]  # Load PCA-reduced feature names

# Function to preprocess input sequence
def encode_sequence(user_sequence):
    """Encodes and pads the user input sequence."""
    encoded_seq = [activity_encoder.get(activity, 0) for activity in user_sequence]
    padded_seq = encoded_seq + [0] * (max_seq_length - len(encoded_seq))
    return np.array(padded_seq).reshape(1, -1)

# Function to extract test values
def extract_test_values(user_sequence):
    """Extracts Leucocytes, CRP, and LacticAcid values if present in input."""
    leucocytes, crp, lactic_acid = np.nan, np.nan, np.nan
    for activity in user_sequence:
        if "Leucocytes(" in activity:
            leucocytes = float(activity.split("(")[-1].strip(")"))
        elif "CRP(" in activity:
            crp = float(activity.split("(")[-1].strip(")"))
        elif "LacticAcid(" in activity:
            lactic_acid = float(activity.split("(")[-1].strip(")"))
    return leucocytes, crp, lactic_acid

# Streamlit Web App
st.title("Sepsis Activity Prediction (PCA-Optimized)")
st.write("Enter a sequence of activities and the necessary patient attributes.")

# User input: Sequence of activities
user_sequence = st.text_input("Enter activities (comma-separated, e.g., 'ER Registration, CRP(160), Leucocytes(12)')")

# User input: Only necessary PCA-reduced features
st.write("Enter key patient attributes (True/False for conditions, Age as a number):")
patient_attributes = []
for attr in pca_feature_names:  # Only take relevant PCA-reduced inputs
    value = st.text_input(f"{attr}", key=attr)
    if value.lower() in ["true", "false"]:
        patient_attributes.append(1 if value.lower() == "true" else 0)
    else:
        patient_attributes.append(float(value) if value else np.nan)

# Predict button
if st.button("Predict Next Activity"):
    user_sequence_list = [x.strip() for x in user_sequence.split(",") if x]
    
    # Extract test values
    leucocytes, crp, lactic_acid = extract_test_values(user_sequence_list)

    # Encode sequence
    encoded_seq = encode_sequence(user_sequence_list)

    # Combine all inputs (only necessary features)
    user_input = np.hstack([encoded_seq, patient_attributes, [leucocytes, crp, lactic_acid]])
    user_input = np.nan_to_num(user_input, nan=0)  # Replace NaNs with 0

    # Predict next activity
    predicted_index = rf_model.predict(user_input.reshape(1, -1))[0]
    predicted_activity = label_classes[predicted_index]

    # Display result
    st.success(f"**Predicted Next Activity: {predicted_activity}**")
