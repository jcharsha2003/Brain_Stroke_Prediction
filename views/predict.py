import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import xgboost as xgb
import json
from streamlit_lottie import st_lottie

# Load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_predict_dark = load_lottiefile("./assets/predict.json")
lottie_predict_light = load_lottiefile("./assets/predicttwo.json")

# Load the models and data
df = pd.read_csv('./data/data_cluster1.csv')

# Define features for clustering and XGBoost
features_for_clustering = [
    'gender', 'age', 'Race', 'Marital status', 'alcohol', 'smoke',
    'sleep disorder', 'General health condition', 'depression',
    'sleep time', 'diabetes', 'hypertension', 'Minutes sedentary activity',
    'Coronary Heart Disease', 'Body Mass Index', 'Waist Circumference',
    'Systolic blood pressure', 'Diastolic blood pressure',
    'High-density lipoprotein', 'Low-density lipoprotein'
]
all_features_for_xgboost = df.columns.tolist()
all_features_for_xgboost.remove('stoke')
features_to_fill = [f for f in all_features_for_xgboost
                   if f not in features_for_clustering and f != 'cluster']

# Prepare data
X_clustering = df[features_for_clustering]
X_xgboost = df[all_features_for_xgboost]
y = df['stoke']

# Initialize and fit scalers
scaler_clustering = StandardScaler()
scaler_clustering.fit(X_clustering)  # Fit the scaler

scaler_xgboost = StandardScaler()
scaler_xgboost.fit(X_xgboost)  # Fit the scaler

# Fit KMeans
kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)

X_clustering_scaled = scaler_clustering.transform(X_clustering)
kmeans.fit(X_clustering_scaled)

# Load the XGBoost model
xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model('./data/xgboost_model.json')

def predict():
    st.title("Stroke Prediction")
    st.subheader("Review your information and make a prediction.")

    # Add a theme selector
    theme = st.selectbox("Please Select ,Selected Theme in Streamlit settings", options=["Dark", "Light"])

    # Get user data from session state
    user_data = st.session_state.get("user_data", {})

    # Define all required fields
    required_fields = [
        'gender', 'age', 'race', 'marital_status', 'alcohol', 'smoke',
        'sleep_disorder', 'general_health_condition', 'depression',
        'sleep_time', 'sedentary_minutes', 'diabetes', 'hypertension',
        'coronary_heart_disease', 'body_mass_index', 'waist_circumference',
        'systolic_blood_pressure', 'diastolic_blood_pressure', 'hdl', 'ldl'
    ]

    # Create a DataFrame for displaying user data
    user_input = {
        'gender': user_data.get('gender', 'Not Filled'),
        'age': user_data.get('age', 'Not Filled'),
        'Race': user_data.get('race', 'Not Filled'),
        'Marital status': user_data.get('marital_status', 'Not Filled'),
        'alcohol': user_data.get('alcohol', 'Not Filled'),
        'smoke': user_data.get('smoke', 'Not Filled'),
        'sleep disorder': user_data.get('sleep_disorder', 'Not Filled'),
        'General health condition': user_data.get('general_health_condition', 'Not Filled'),
        'depression': user_data.get('depression', 'Not Filled'),
        'sleep time': user_data.get('sleep_time', 'Not Filled'),
        'diabetes': user_data.get('diabetes', 'Not Filled'),
        'hypertension': user_data.get('hypertension', 'Not Filled'),
        'Minutes sedentary activity': user_data.get('sedentary_minutes', 'Not Filled'),
        'Coronary Heart Disease': user_data.get('coronary_heart_disease', 'Not Filled'),
        'Body Mass Index': user_data.get('body_mass_index', 'Not Filled'),
        'Waist Circumference': user_data.get('waist_circumference', 'Not Filled'),
        'Systolic blood pressure': user_data.get('systolic_blood_pressure', 'Not Filled'),
        'Diastolic blood pressure': user_data.get('diastolic_blood_pressure', 'Not Filled'),
        'High-density lipoprotein': user_data.get('hdl', 'Not Filled'),
        'Low-density lipoprotein': user_data.get('ldl', 'Not Filled')
    }

    df_user_input = pd.DataFrame(user_input.items(), columns=['Feature', 'Value'])

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Display the user data table
        st.subheader("Your Information")
        st.dataframe(df_user_input)

    with col2:
        # Display the Lottie animation based on the selected theme
        st_lottie(
            lottie_predict_dark if theme == "Dark" else lottie_predict_light,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            height=220,  # Adjust height
            width=270,   # Adjust width
            key="predict_animation"
        )

    # Predict button
    if st.button("Predict"):
        # Check for missing fields
        missing_fields = [field for field in required_fields if user_data.get(field, 'Not Filled') == 'Not Filled']

        if not missing_fields:
            # If all fields are filled, proceed with prediction
            st.success("All fields are filled. Making prediction...")

            # Prepare input for prediction
            input_df = pd.DataFrame([user_input])
            
            # Predict cluster
            input_clustering_scaled = scaler_clustering.transform(input_df[features_for_clustering])
            cluster_label = kmeans.predict(input_clustering_scaled)[0]
            input_df['cluster'] = cluster_label

            # Find nearest neighbors and fill missing features
            cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
            cluster_data_scaled = scaler_clustering.transform(df.iloc[cluster_indices][features_for_clustering])

            knn = NearestNeighbors(n_neighbors=5)
            knn.fit(cluster_data_scaled)
            distances, indices = knn.kneighbors(input_clustering_scaled)

            nearest_record_index = cluster_indices[indices[0][0]]
            nearest_record = df.iloc[nearest_record_index]

            for feature in features_to_fill:
                input_df[feature] = nearest_record[feature]

            # Predict stroke
            input_xgboost = input_df[all_features_for_xgboost]
            input_xgboost_scaled = scaler_xgboost.transform(input_xgboost)
            stroke_prediction = xgboost_model.predict(input_xgboost_scaled)[0]
            st.write(f"Predicted Stroke Risk: {'Yes' if stroke_prediction == 1 else 'No'}")
        else:
            # Notify user which fields are missing
            st.error(f"Please fill out the following fields on the previous pages: {', '.join(missing_fields)}")

if __name__ == "__main__":
    predict()
