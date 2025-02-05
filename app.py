import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# ✅ Biomarker-Based Decision Rules
biomarker_priority = ["LacticAcid", "CRP", "Leucocytes"]
biomarker_next_activity_mapping = {
    "Leucocytes": {"High": "LacticAcid", "Elevated": "CRP", "Normal": "ER Triage"},
    "CRP": {"Severe": "IV Antibiotics", "Moderate": "LacticAcid", "Low": "ER Triage"},
    "LacticAcid": {"Critical": "ICU Admission", "High": "IV Fluid", "Normal": "ER Triage"}
}

# ✅ Biomarker Options
biomarker_options = {
    "Leucocytes": ["NaN", "Low (0-7.5)", "Normal (7.5-12.5)", "Elevated (12.5-15.0)", "High (15.0-30.0)", "Critical (>30.0)"],
    "CRP": ["NaN", "Low (0-50)", "Mild (50-100)", "Moderate (100-150)", "Severe (150-250)", "Critical (>250)"],
    "LacticAcid": ["NaN", "Normal (0-1.2)", "Borderline (1.2-1.8)", "Elevated (1.8-2.5)", "High (2.5-4.0)", "Critical (>4.0)"]
}

def clean_biomarker_value(value):
    return None if value == "NaN" else value

# ✅ Predict Next Activity & Remaining Time
def predict_next_activity_and_time(activity_sequence, feature_values, biomarker_values):
    sequence = tokenizer.texts_to_sequences([activity_sequence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')

    feature_df = pd.DataFrame([feature_values], columns=[
        "SIRSCriteria2OrMore", "Infusion", "SIRSCritTemperature", "DiagnosticLacticAcid",
        "SIRSCritHeartRate", "DiagnosticXthorax", "SIRSCritTachypnea",
        "DiagnosticUrinarySediment", "Age", "InfectionSuspected"
    ])
    feature_array = scaler.transform(feature_df)

    final_biomarkers = {}
    for biomarker, value in biomarker_values.items():
        final_biomarkers[biomarker] = value

    for biomarker in biomarker_priority:
        if biomarker in final_biomarkers and final_biomarkers[biomarker] in biomarker_next_activity_mapping[biomarker]:
            return biomarker_next_activity_mapping[biomarker][final_biomarkers[biomarker]], 0

    model_prediction = model.predict([padded_sequence, feature_array])
    predicted_class = np.argmax(model_prediction, axis=1)
    predicted_activity = label_encoder.inverse_transform(predicted_class)[0]

    predicted_activity_duration = df_time_avg[df_time_avg["Activity"] == predicted_activity]["Avg Remaining Time"].values[0] if predicted_activity in df_time_avg["Activity"].values else 600
    predicted_remaining_time = time_model.predict([[predicted_activity_duration]])[0]
    
    return predicted_activity, round(predicted_remaining_time, 2)

# ✅ Streamlit UI
st.title("Sepsis Next Activity & Time Prediction")
st.subheader("Enter Patient Details")

# ✅ Activity Sequence Selection
st.write("### Activity Sequence")
all_activities = [
    "Leucocytes", "CRP", "LacticAcid", "ER Triage", "ER Sepsis Triage", "IV Liquid", "IV Antibiotics", 
    "Admission NC", "Release A", "Return ER", "Admission IC", "Release B", "Release C", "Release D", "Release E"
]
if "activity_list" not in st.session_state:
    st.session_state.activity_list = ["ER Registration"]
st.write("Activity 1: ER Registration")

selected_biomarkers = []
for i in range(1, len(st.session_state.activity_list)):
    col1, col2 = st.columns([4, 1])
    with col1:
        activity = st.selectbox(f"Activity {i+1}", all_activities, key=f"activity_{i}")
        st.session_state.activity_list[i] = activity
        if activity in biomarker_priority:
            selected_biomarkers.append((activity, i))
    with col2:
        if st.button(f"❌", key=f"remove_{i}"):
            st.session_state.activity_list.pop(i)
            st.rerun()

if st.button("+ Add Activity"):
    st.session_state.activity_list.append(all_activities[0])
    st.rerun()

final_activity_sequence = " -> ".join(st.session_state.activity_list)

# ✅ Biomarker Levels (Allow multiple entries if repeated)
st.write("### Biomarker Levels")
biomarker_values = {}
for biomarker, index in selected_biomarkers:
    biomarker_values[(biomarker, index)] = st.selectbox(
        f"{biomarker} Level (Activity {index+1})", 
        biomarker_options[biomarker], 
        key=f"biomarker_{biomarker}_{index}"
    )

# ✅ Patient Features
st.write("### Patient Features")
feature_values = []
feature_labels = [
    "SIRSCriteria2OrMore", "Infusion", "SIRSCritTemperature", "DiagnosticLacticAcid",
    "SIRSCritHeartRate", "DiagnosticXthorax", "SIRSCritTachypnea",
    "DiagnosticUrinarySediment", "Age", "InfectionSuspected"
]

cols = st.columns(3)  # Split features into 4 columns
for i, label in enumerate(feature_labels):
    with cols[i % 3]:
        feature_values.append(st.checkbox(label, value=True) if label != "Age" else st.slider(label, 5, 110, 30))

# ✅ Predict Button
if st.button("Predict Next Activity & Time"):
    last_activity = st.session_state.activity_list[-1]  # Get the last activity in the sequence
    
    if last_activity in ["Release A", "Release B", "Release C", "Release D", "Release E", "Return ER"]:
        st.success("Process Completed ✅")
        st.info("Estimated Remaining Time: 0 seconds")
    else:
        predicted_activity, predicted_time = predict_next_activity_and_time(final_activity_sequence, feature_values, biomarker_values)
        st.success(f"Predicted Next Activity: {predicted_activity}")
        st.info(f"Estimated Remaining Time: {predicted_time} seconds (~{predicted_time/3600:.2f} hours)")
