import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestRegressor

# ✅ Load all saved models and preprocessors
model = load_model("sepsis_lstm_model.keras")

with open("sepsis_tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("sepsis_label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

with open("sepsis_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("sepsis_time_model.pkl", "rb") as file:
    time_model = pickle.load(file)

# ✅ Load Average Remaining Time Data
df_time_avg = pd.read_csv("Sepsis_Avg_Activity_Duration.csv")

# ✅ Define input shape
max_sequence_length = model.input_shape[0][1]
feature_input_shape = model.input_shape[1][1]

# ✅ Biomarker-Based Decision Rules (Correct Biomarker Labels)
biomarker_priority = ["LacticAcid", "CRP", "Leucocytes"]
biomarker_next_activity_mapping = {
    "Leucocytes": {"High": "LacticAcid", "Elevated": "CRP", "Normal": "ER Triage"},
    "CRP": {"Severe": "IV Antibiotics", "Moderate": "LacticAcid", "Low": "ER Triage"},
    "LacticAcid": {"Critical": "ICU Admission", "High": "IV Fluid", "Normal": "ER Triage"}
}

# ✅ Correct Biomarker Dropdown Options
biomarker_options = {
    "Leucocytes": ["NaN", "Low (0-7.5)", "Normal (7.5-12.5)", "Elevated (12.5-15.0)", "High (15.0-30.0)", "Critical (>30.0)"],
    "CRP": ["NaN", "Low (0-50)", "Mild (50-100)", "Moderate (100-150)", "Severe (150-250)", "Critical (>250)"],
    "LacticAcid": ["NaN", "Normal (0-1.2)", "Borderline (1.2-1.8)", "Elevated (1.8-2.5)", "High (2.5-4.0)", "Critical (>4.0)"]
}

def clean_biomarker_value(value):
    return None if value == "NaN" else value.split(" ")[0]  # Treat "NaN" as missing


# ✅ Function to Predict Next Activity
def predict_next_activity(activity_sequence, feature_values, biomarker_values):
    sequence = tokenizer.texts_to_sequences([activity_sequence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')

    feature_df = pd.DataFrame([feature_values], columns=[
        "DiagnosticArtAstrup", "DiagnosticUrinarySediment", "SIRSCritHeartRate", "SIRSCritTachypnea",
        "SIRSCritTemperature", "Hypotensie", "SIRSCritLeucos", "DiagnosticLacticAcid", "Oligurie",
        "Hypoxie", "DisfuncOrg", "Infusion", "Age", "InfectionSuspected"
    ])
    feature_array = scaler.transform(feature_df)

    # ✅ Check Biomarker-Based Rules First
    for biomarker in biomarker_priority:
        cleaned_value = biomarker_values[biomarker]
        if cleaned_value is not None and cleaned_value in biomarker_next_activity_mapping[biomarker]:
            return biomarker_next_activity_mapping[biomarker][cleaned_value]


    # ✅ Predict Next Activity
    model_prediction = model.predict([padded_sequence, feature_array])
    predicted_class = np.argmax(model_prediction, axis=1)
    decoded_prediction = label_encoder.inverse_transform(predicted_class)[0]

    return decoded_prediction

# ✅ Function to Predict Next Activity and Remaining Time
def predict_next_activity_and_time(activity_sequence, feature_values, biomarker_values):
    predicted_next_activity = predict_next_activity(activity_sequence, feature_values, biomarker_values)

    if predicted_next_activity in df_time_avg["Activity"].values:
        predicted_activity_duration = df_time_avg[df_time_avg["Activity"] == predicted_next_activity]["Avg Remaining Time"].values[0]
    else:
        predicted_activity_duration = 600  # Default 10 minutes

    predicted_remaining_time = time_model.predict([[predicted_activity_duration]])[0]
    return predicted_next_activity, round(predicted_remaining_time, 2)

# ✅ Streamlit UI
st.title("Sepsis Next Activity & Time Prediction")
st.subheader("Enter Patient Details")

# ✅ Activity Sequence Selection (Drop-downs + Add/Remove)
st.write("### Activity Sequence")

all_activities = [
    "Leucocytes", "CRP", "LacticAcid", "ER Triage", "ER Sepsis Triage", "IV Liquid", "IV Antibiotics", 
    "Admission NC", "Release A", "Return ER", "Admission IC", "Release B", "Release C", "Release D", "Release E"
]

if "activity_list" not in st.session_state:
    st.session_state.activity_list = ["ER Registration"]

st.write("Activity 1: ER Registration")

for i in range(1, len(st.session_state.activity_list)):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state.activity_list[i] = st.selectbox(f"Activity {i+1}", all_activities, 
                                                         index=all_activities.index(st.session_state.activity_list[i]) if st.session_state.activity_list[i] in all_activities else 0)
    with col2:
        if st.button(f"❌", key=f"remove_{i}"):
            st.session_state.activity_list.pop(i)
            st.rerun()



if st.button("+ Add Activity"):
    st.session_state.activity_list.append(all_activities[0])
    st.rerun()



final_activity_sequence = " -> ".join(st.session_state.activity_list)

# ✅ Patient Features (6 columns)
st.write("### Patient Features")
cols = st.columns(3)
feature_values = []
feature_labels = [
    "DiagnosticArtAstrup", "DiagnosticUrinarySediment", "SIRSCritHeartRate", "SIRSCritTachypnea",
    "SIRSCritTemperature", "Hypotensie", "SIRSCritLeucos", "DiagnosticLacticAcid", "Oligurie",
    "Hypoxie", "DisfuncOrg", "Infusion", "Age", "InfectionSuspected"
]

for i, label in enumerate(feature_labels):
    with cols[i % 3]:
        if label == "Age":
            feature_values.append(st.slider("Age", min_value=5, max_value=110, value=30))
        else:
            feature_values.append(1 if st.radio(f"{label}", ["False", "True"]) == "True" else 0)

# ✅ Biomarker Selection (With Full Options)
st.write("### Biomarker Levels")
biomarker_values = {}
for biomarker in biomarker_priority:
    biomarker_values[biomarker] = st.selectbox(f"{biomarker} Level", biomarker_options[biomarker])

# ✅ Predict Button
if st.button("Predict Next Activity & Time"):
    # Convert NaN to Normal
    cleaned_biomarkers = {k: clean_biomarker_value(v) for k, v in biomarker_values.items()}
    
    predicted_activity, predicted_time = predict_next_activity_and_time(final_activity_sequence, feature_values, cleaned_biomarkers)
    
    st.success(f"Predicted Next Activity: {predicted_activity}")
    st.info(f"Estimated Remaining Time: {predicted_time} seconds (~{predicted_time/3600:.2f} hours)")
