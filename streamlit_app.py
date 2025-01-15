import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_path = "data/Classification.csv"
data = pd.read_csv(data_path)

# Set page config
st.set_page_config(page_title="Drug Classification App", layout="wide", page_icon="💊")

# Custom CSS for enhanced appearance
st.markdown(
    """
    <style>
    .main {
        background-color: #e8f4f8;
        font-family: Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton > button {
        background-color: #1abc9c;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #16a085;
        color: #f1f1f1;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 20px;
    }
    .metric {
        background-color: #ffffff;
        border: 2px solid #1abc9c;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("💊 Drug Classification App")
st.write("### Classify drug types based on patient data using K-Nearest Neighbors (KNN).")

# Preprocessing (outside navigation to ensure availability)
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
