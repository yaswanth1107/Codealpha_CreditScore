import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- App Configuration and Styling ---

def set_background():
    \"\"\"
    Applies custom CSS to set a background image and style text elements for better visibility.
    The CSS targets the main app container and specific Streamlit components.
    \"\"\"
    st.markdown(
        '''
        <style>
        /* Main app background */
        .stApp {
            background-image: url("https://media.istockphoto.com/id/1449244963/photo/prospect-exchange-rate-amid-financial-crisis.jpg?s=2048x2048&w=is&k=20&c=I_Zi0BDaN33Rgy9rfhWdoW8I4qoxJWa3KedekGMowRs=");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Styling for text elements to ensure they are white and readable against the dark background */
        .stMarkdown, .stTextInput > label, .stNumberInput > label, .css-1cpxqw2, .css-qrbaxs, .css-10trblm {
            color: white !important;
        }
        
        /* Specific styling for a title class if you need black text */
        h1.title-black {
            color: black !important;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

# Set the page configuration. This should be the first Streamlit command.
st.set_page_config(page_title="Credit Scoring Predictor", layout="centered")

# Apply the custom background style
set_background()

# --- Model Loading and Caching ---

@st.cache_resource
def load_model():
    \"\"\"
    Loads the dataset, trains a RandomForest model, and fits a scaler.
    Uses @st.cache_resource to load and train the model only once, speeding up the app on subsequent runs.
    
    Returns:
        tuple: A tuple containing the trained model, the fitted scaler, and the feature column names.
    \"\"\"
    # Load the dataset from the CSV file
    # Ensure "UCI_Credit_Card.csv" is in the same directory as the script.
    try:
        df = pd.read_csv(r"C:\\Users\\Yaswanth\\Downloads\\UCI_Credit_Card.csv")
    except FileNotFoundError:
        st.error("Error: 'UCI_Credit_Card.csv' not found. Please make sure the dataset file is in the correct directory.")
        return None, None, None

    # Define features (X) and target (y)
    X = df.drop(columns=["ID", "default.payment.next.month"])
    y = df["default.payment.next.month"]
    
    # Initialize and fit the StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

# --- Prediction Function ---

def predict(model, scaler, user_input):
    \"\"\"
    Takes user input, scales it, and uses the trained model to make a prediction.
    
    Args:
        model (RandomForestClassifier): The trained machine learning model.
        scaler (StandardScaler): The fitted scaler object.
        user_input (dict): A dictionary containing the user's input values for each feature.
        
    Returns:
        tuple: A tuple containing the prediction (0 or 1) and the probability of default.
    \"\"\"
    # Convert the user input dictionary to a DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Scale the user's input using the pre-fitted scaler
    user_scaled = scaler.transform(user_df)
    
    # Make a prediction
    prediction = model.predict(user_scaled)[0]
    
    # Calculate the prediction probability (probability of the positive class '1')
    probability = model.predict_proba(user_scaled)[0][1]
    
    return prediction, probability

# --- Streamlit UI ---

# Display the main title and subtitle of the application
st.markdown('<h1 style="color:white;">💳 Creditworthiness Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:white;">📥 Enter Financial Details</h3>', unsafe_allow_html=True)

# Load the model, scaler, and feature names
model, scaler, feature_names = load_model()

# Only proceed if the model loaded successfully
if model and scaler and feature_names is not None:
    # Create a dictionary to hold the user's input
    user_input = {}
    
    # Dynamically create number input fields for each feature
    for feature in feature_names:
        # Use a more readable label by replacing underscores with spaces and capitalizing
        label = feature.replace('_', ' ').title()
        user_input[feature] = st.number_input(label, value=0.0, format="%.2f")

    # Create the prediction button
    if st.button("🔮 Predict Creditworthiness"):
        # Call the predict function with the model, scaler, and user's input
        pred, prob = predict(model, scaler, user_input)
        
        # Display the result based on the prediction
        if pred == 1:
            st.error(f"❌ This person is likely to default! (Risk Score: {prob:.2f})")
        else:
            st.success(f"✅ This person is likely creditworthy! (Risk Score: {prob:.2f})")
