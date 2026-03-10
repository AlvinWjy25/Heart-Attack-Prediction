import streamlit as st
import joblib
import pandas as pd
import mlflow.sklearn
import os
from train import run_training

# Setup MLflow Tracking
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# Setup Session State for the RUN_ID
if 'run_id' not in st.session_state:
    st.session_state.run_id = "8a41522603e44554b41d706f701188d8" 

# Load the model dynamically based on the current RUN_ID
# @st.cache_resource
# def load_model(run_id):
#     model_uri = f"runs:/{run_id}/model"
#     return mlflow.sklearn.load_model(model_uri)

@st.cache_resource #Streamlit pickl caching
def load_model():
    model_path = os.path.join(BASE_DIR, "artifacts", "heart_attack_pipeline.pkl")
    return joblib.load(model_path)

def main():
    st.title('🫀 Heart Attack Prediction App')
    st.write("Enter the patient's medical details below to predict the likelihood of a heart attack.")

    # Load the pipeline
    pipeline = load_model()

    # INPUT COLUMNS
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=20, max_value=100, value=50)
        
        sex_map = {1: '1: Male', 0: '0: Female'}
        sex = st.selectbox('Sex', options=list(sex_map.keys()), format_func=lambda x: sex_map[x])
        
        cp_map = {0: '0: Typical Angina', 1: '1: Atypical Angina', 2: '2: Non-anginal Pain', 3: '3: Asymptomatic'}
        cp = st.selectbox('Chest Pain Type (cp)', options=list(cp_map.keys()), format_func=lambda x: cp_map[x])
        
        trestbps = st.number_input('Resting Blood Pressure [mm Hg] (trestbps)', min_value=80, max_value=200, value=120)
        
        chol = st.number_input('Serum Cholesterol [mg/dl] (chol)', min_value=100, max_value=600, value=200)

    with col2:
        fbs_map = {0: '0: False (<= 120 mg/dl)', 1: '1: True (> 120 mg/dl)'}
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=list(fbs_map.keys()), format_func=lambda x: fbs_map[x])
        
        restecg_map = {0: '0: Normal', 1: '1: ST-T wave abnormality', 2: '2: Left ventricular hypertrophy'}
        restecg = st.selectbox('Resting ECG (restecg)', options=list(restecg_map.keys()), format_func=lambda x: restecg_map[x])
        
        thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
        
        exang_map = {0: '0: No', 1: '1: Yes'}
        exang = st.selectbox('Exercise Induced Angina (exang)', options=list(exang_map.keys()), format_func=lambda x: exang_map[x])

    with col3:
        oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        slope_map = {0: '0: Upsloping', 1: '1: Flat', 2: '2: Downsloping'}
        slope = st.selectbox('Slope of Peak Exercise ST Segment (slope)', options=list(slope_map.keys()), format_func=lambda x: slope_map[x])
        
        ca_map = {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels', 4: '4 (null)'}
        ca = st.selectbox('Major Vessels Colored by Flourosopy (ca)', options=list(ca_map.keys()), format_func=lambda x: ca_map[x])
        
        thal_map = {0: '0: Null', 1: '1: Normal', 2: '2: Fixed Defect', 3: '3: Reversable Defect'}
        thal = st.selectbox('Thalassemia (thal)', options=list(thal_map.keys()), format_func=lambda x: thal_map[x])
    
    # PREDICTION LOGIC
    if st.button('Make Prediction', type="secondary", use_container_width=True):
        input_data = {
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
            'ca': [ca], 'thal': [thal]
        }
        
        df = pd.DataFrame(input_data)
        prediction = pipeline.predict(df)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.error('⚠️ **Prediction: High Risk of Heart Attack**')
            st.write("Based on the provided metrics, the model indicates a higher likelihood of cardiovascular presence. Please consult a medical professional.")
        else:
            st.success('✅ **Prediction: Low Risk of Heart Attack**')
            st.write("Based on the provided metrics, the model indicates a lower likelihood of cardiovascular presence.")

if __name__ == '__main__':
    main()