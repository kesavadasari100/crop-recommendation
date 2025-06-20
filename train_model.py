import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the data
data = pd.read_csv("crop_data.csv")

# Encode the Soil_Type column
le_soil = LabelEncoder()
data['Soil_Type'] = le_soil.fit_transform(data['Soil_Type'])

# Encode the Crop (Target)
le_crop = LabelEncoder()
data['Crop'] = le_crop.fit_transform(data['Crop'])

# Save both encoders for future use
with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(le_soil, f)

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

# Features and Target
X = data[['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Soil_Type']]
y = data['Crop']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")