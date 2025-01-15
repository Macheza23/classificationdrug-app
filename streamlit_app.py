import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set page config with custom icon (upload the icon image to your app folder)
st.set_page_config(
    page_title="Drug Classification App", 
    layout="wide", 
    page_icon="new_icon.png"  # Replace with the path to your custom icon
)

# Custom CSS for a modern, vibrant, and user-friendly look
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin: 2rem 0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333333;
        font-weight: bold;
        text-align: center;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 20px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .stSlider > div {
        margin: 10px 0;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 20px;
    }
    .metric {
        background-color: #1abc9c;
        color: white;
        border-radius: 12px;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    .sidebar .sidebar-content .stButton > button {
        background-color: #e67e22;
        border-radius: 12px;
        padding: 10px 20px;
    }
    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
        width: 100%;
    }
    .stSelectbox, .stNumberInput {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("💊 Drug Classification App")
st.write("### Classify drug types based on patient data using K-Nearest Neighbors (KNN).")

# Load the dataset
data = None
data_path = "Classification.csv"

# First, attempt to load the CSV from the given path
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.warning("Dataset 'Classification.csv' not found.")
    
    # Allow the user to upload a new dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

if data is not None:
    # Preprocessing
    label_encoders = {}
    encoded_data = data.copy()
    for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        le = LabelEncoder()
        encoded_data[column] = le.fit_transform(encoded_data[column])
        label_encoders[column] = le

    # Splitting data
    X = encoded_data.drop('Drug', axis=1)
    y = encoded_data['Drug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to:", ["Dataset Overview", "Model Training", "Make a Prediction"])

    if page == "Dataset Overview":
        st.header("Dataset Overview")
        # Show dataset
        if st.checkbox("Show dataset preview"):
            st.write(data.head())

        # Show data statistics
        if st.checkbox("Show dataset statistics"):
            st.write(data.describe(include='all'))

    elif page == "Model Training":
        st.header("Model Training")

        # Model training
        k = st.slider("Select the number of neighbors (k)", min_value=1, max_value=15, value=5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Results
        st.subheader("Model Performance")
        st.markdown(
            f"""
            <div class='metric-container'>
                <div class='metric'>Accuracy: {accuracy * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Classification report
        if st.checkbox("Show classification report"):
            report = classification_report(y_test, y_pred, target_names=label_encoders['Drug'].classes_)
            st.text(report)

    elif page == "Make a Prediction":
        st.header("Make a Prediction")
        st.write("Enter patient details below to predict the appropriate drug.")

        # User input for prediction
        with st.form("prediction_form"):
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            sex = st.selectbox("Sex", label_encoders['Sex'].classes_)
            bp = st.selectbox("Blood Pressure (BP)", label_encoders['BP'].classes_)
            cholesterol = st.selectbox("Cholesterol", label_encoders['Cholesterol'].classes_)
            na_to_k = st.number_input("Na to K ratio", min_value=0.0, max_value=50.0, value=15.0)
            submitted = st.form_submit_button("Predict Drug")

        if submitted:
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [label_encoders['Sex'].transform([sex])[0]],
                'BP': [label_encoders['BP'].transform([bp])[0]],
                'Cholesterol': [label_encoders['Cholesterol'].transform([cholesterol])[0]],
                'Na_to_K': [na_to_k]
            })
            model = KNeighborsClassifier(n_neighbors=5)  # Default model in case training not run
            model.fit(X_train, y_train)
            prediction = model.predict(input_data)
            predicted_drug = label_encoders['Drug'].inverse_transform(prediction)[0]
            st.success(f"Predicted Drug: {predicted_drug}")
else:
    st.error("No dataset loaded. Please upload a dataset to continue.")
