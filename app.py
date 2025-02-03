import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models and preprocessors
with open("sepsis_lstm_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("sepsis_tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("sepsis_label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

with open("sepsis_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("sepsis_time_model.pkl", "rb") as file:
    time_model = pickle.load(file)

# Define necessary variables
biomarker_priority = ["LacticAcid", "CRP", "Leucocytes"]
biomarker_next_activity_mapping = {
    "Leucocytes": {"High": "LacticAcid", "Elevated": "CRP", "Normal": "ER Triage"},
    "CRP": {"Severe": "IV Antibiotics", "Moderate": "LacticAcid", "Low": "ER Triage"},
    "LacticAcid": {"Critical": "ICU Admission", "High": "IV Fluid", "Normal": "ER Triage"}
}
max_sequence_length = 10  # Adjust based on training data

# Function to predict next activity and remaining time
def predict_next_activity_and_time(activity_sequence, feature_values, biomarker_values):
    sequence = tokenizer.texts_to_sequences([activity_sequence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    
    feature_array = np.array(feature_values).reshape(1, -1)
    feature_array = scaler.transform(feature_array)
    
    for biomarker in biomarker_priority:
        if biomarker in biomarker_values:
            biomarker_value = biomarker_values[biomarker]
            if biomarker_value in biomarker_next_activity_mapping[biomarker]:
                return biomarker_next_activity_mapping[biomarker][biomarker_value], 3600  # Default 1-hour time estimate
    
    model_prediction = model.predict([padded_sequence, feature_array])
    predicted_class = np.argmax(model_prediction, axis=1)
    predicted_next_activity = label_encoder.inverse_transform(predicted_class)[0]
    
    if predicted_next_activity in ["Release A", "Release B", "Release C", "Release D", "Release E", "Return ER"]:
        predicted_remaining_time = 0
    else:
        predicted_remaining_time = time_model.predict([[600]])[0]  # Default 10 minutes
    
    return predicted_next_activity, round(predicted_remaining_time, 2)

# Streamlit UI
st.title("Sepsis Next Activity & Time Prediction")
st.subheader("Enter Patient Details and Biomarker Levels")

all_activities = ["Leucocytes", "CRP", "LacticAcid", "ER Triage",
       "ER Sepsis Triage", "IV Liquid", "IV Antibiotics", "Admission NC",
       "Release A", "Return ER", "Admission IC", "Release B", "Release C",
       "Release D", "Release E"]
st.write("### Activity Sequence")

# Initialize session state for activities if not already present
if "activity_list" not in st.session_state:
    st.session_state.activity_list = ["ER Registration"]

# Display first activity as fixed "ER Registration"
st.write("Activity 1: ER Registration")

# Display existing activities as dropdowns
for i in range(1, len(st.session_state.activity_list)):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state.activity_list[i] = st.selectbox(f"Activity {i+1}", all_activities, index=all_activities.index(st.session_state.activity_list[i]) if st.session_state.activity_list[i] in all_activities else 0)
    with col2:
        if st.button(f"❌", key=f"remove_{i}"):
            st.session_state.activity_list.pop(i)
            st.rerun()

# Add activity button
if st.button("+ Add Activity"):
    st.session_state.activity_list.append(all_activities[0])  # Default new activity as "Leucocytes"

final_activity_sequence = " -> ".join(st.session_state.activity_list)

st.write("### Patient Features")
col1, col2 = st.columns(2)
feature_values = []
feature_labels = [
    "DiagnosticArtAstrup", "DiagnosticUrinarySediment", "SIRSCritHeartRate", "SIRSCritTachypnea",
    "SIRSCritTemperature", "Hypotensie", "SIRSCritLeucos", "DiagnosticLacticAcid", "Oligurie",
    "Hypoxie", "DisfuncOrg", "Infusion", "Age", "InfectionSuspected"
]
for index, label in enumerate(feature_labels):
    with col1 if index % 2 == 0 else col2:
        if label == "Age":
            feature_values.append(st.slider("Age", min_value=5, max_value=110, value=30))
        else:
            feature_values.append(1 if st.radio(f"{label}", ["False", "True"]) == "True" else 0)

biomarker_values = {}
biomarkers = ["Leucocytes", "CRP", "LacticAcid"]
biomarker_options = {
    "Leucocytes": ["NaN", "Low (0-7.5)", "Normal (7.5-12.5)", "Elevated (12.5-15.0)", "High (15.0-30.0)", "Critical (>30.0)"],
    "CRP": ["NaN", "Low (0-50)", "Mild (50-100)", "Moderate (100-150)", "Severe (150-250)", "Critical (>250)"],
    "LacticAcid": ["NaN", "Normal (0-1.2)", "Borderline (1.2-1.8)", "Elevated (1.8-2.5)", "High (2.5-4.0)", "Critical (>4.0)"]
}
for biomarker in biomarkers:
    biomarker_values[biomarker] = st.selectbox(f"{biomarker} Level", biomarker_options[biomarker])
    if biomarker_values[biomarker] == "NaN":
        biomarker_values[biomarker] = "Normal"  # Default to Normal if NaN is selected

if st.button("Predict Next Activity & Time"):
    predicted_activity, predicted_time = predict_next_activity_and_time(final_activity_sequence, feature_values, biomarker_values)
    st.success(f"Predicted Next Activity: {predicted_activity}")
    st.info(f"Estimated Remaining Time: {predicted_time} seconds (~{predicted_time/3600:.2f} hours)")