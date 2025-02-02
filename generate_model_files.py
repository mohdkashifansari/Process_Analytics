import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load datasets
file_paths = {
    "Sepsis_Activity_Flow": "Sepsis_Activity_Flow.csv",
    "Sepsis_Other_Attributes": "Sepsis_Other_Attributes.csv"
}
datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Extract sequences and attributes
df_sequences = datasets["Sepsis_Activity_Flow"].copy()
df_reduced_attributes = datasets["Sepsis_Other_Attributes"].drop(columns=["Case ID"])  # Remove identifier

# Prepare activity sequence encoding
activity_cols = [col for col in df_sequences.columns if "Activity" in col and col != "Final Activity"]
df_sequences["Activity_Sequence"] = df_sequences[activity_cols].apply(lambda row: [act for act in row if isinstance(act, str)], axis=1)

activity_set = set(activity for seq in df_sequences["Activity_Sequence"] for activity in seq)
activity_encoder = {activity: idx for idx, activity in enumerate(activity_set, start=1)}

sequences_encoded = [[activity_encoder[activity] for activity in seq] for seq in df_sequences["Activity_Sequence"]]
max_seq_length = max(len(seq) for seq in sequences_encoded)
X_sequences_padded = np.array([seq + [0] * (max_seq_length - len(seq)) for seq in sequences_encoded])

# Encode final activity labels
le_activities = LabelEncoder()
y_activities = le_activities.fit_transform(df_sequences["Final Activity"])

# Select the 9 PCA features explicitly
pca_features_names = [
    "DisfuncOrg", "SIRSCritTachypnea", "Hypotensie", "SIRSCritHeartRate",
    "Infusion", "DiagnosticArtAstrup", "Age", "DiagnosticSputum", "SIRSCritTemperature"
]
df_pca_features = df_reduced_attributes[pca_features_names]

# Convert categorical attributes to numeric
df_combined = pd.DataFrame(np.hstack([X_sequences_padded, df_pca_features]))
df_combined_numeric = df_combined.apply(pd.to_numeric, errors='coerce')
df_combined_cleaned = df_combined_numeric.fillna(df_combined_numeric.median())

# Perform PCA to reduce dimensionality to 9 components
num_pca_components = 9
pca = PCA(n_components=num_pca_components)
X_pca = pca.fit_transform(df_combined_cleaned)

# Save PCA-transformed features names
pca_transformed_names = [f"PCA_{name}" for name in pca_features_names]

# Train Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_activities, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Save trained Random Forest model
model_filename = "sepsis_activity_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(rf_model, f)

# Save column information for later use in the web app
columns_filename = "columns.json"
column_data = {
    "max_seq_length": max_seq_length,  # Sequence padding length
    "activity_encoder": activity_encoder,  # Activity encoding dictionary
    "label_classes": le_activities.classes_.tolist(),  # Mapping of labels
    "pca_features": pca_transformed_names  # Only store PCA-reduced feature names
}

with open(columns_filename, "w") as f:
    json.dump(column_data, f)

print(f"✅ Model saved as {model_filename}")
print(f"✅ Column information saved as {columns_filename}")
