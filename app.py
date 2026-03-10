import streamlit as st
import joblib
import pandas as pd
import mlflow.sklearn
import os

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
        proba = pipeline.predict_proba(df)[0]

        low_risk_pct  = proba[0] * 100
        high_risk_pct = proba[1] * 100
        
        st.markdown("---")
        if prediction == 1:
            st.error('⚠️ **Prediction: High Risk of Heart Attack**')
            st.write("Based on the provided metrics, the model indicates a higher likelihood of cardiovascular presence. Please consult a medical professional.")
        else:
            st.success('✅ **Prediction: Low Risk of Heart Attack**')
            st.write("Based on the provided metrics, the model indicates a lower likelihood of cardiovascular presence.")

        # Confidence section
        st.markdown("#### Model Confidence")
        col_low, col_high = st.columns(2)
        with col_low:
            st.metric("✅ Low Risk", f"{low_risk_pct:.1f}%")
            st.progress(proba[0])
        with col_high:
            st.metric("⚠️ High Risk", f"{high_risk_pct:.1f}%")
            st.progress(proba[1])

def run_training():
    with mlflow.start_run(run_name="xgb_heart_fixed_pipeline") as run:

        path = os.path.join(os.getcwd(), "data", "Heart Attack Data Set.csv")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        data = load_data(path) 
        dataset = mlflow.data.from_pandas(data, source=path, name="heart_attack_training")
        mlflow.log_input(dataset, context = "training")

        # Logging data dimension
        mlflow.log_param("data_rows", data.shape[0])
        mlflow.log_param("data_cols", data.shape[1])

        # Logging data artifact
        data.head(10).to_csv("data_sample.csv", index=False) 
        mlflow.log_artifact("data_sample.csv")

        # Logging dataset hash
        mlflow.log_param("data_hash", get_file_hash(path))

        X = data.drop("target", axis=1)
        y = data["target"]

        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        num_cols = X.select_dtypes("number").columns
        cat_cols = X.select_dtypes("object").columns

        # Preprocessing
        print(f"2. [📩] Importing Preprocessor...")
        preprocessor = build_preprocessor(num_cols, cat_cols) 

        # Model Pipeline 
        # Karena parameter sudah ada di dalam pipeline.py, kita set params kosong
        # agar tidak error saat masuk ke fungsi build_pipeline(preprocessor, params)
        
        print("3. [⚙️] Using Hardcoded Parameters from pipeline.py...")
        params = {'n_estimators': 418, 
            'max_depth': 5, 
            'learning_rate': 0.16002139803210555, 
            'subsample': 0.6458307113580525, 
            'colsample_bytree': 0.8072537753178493, 
            'gamma': 1.989869515236475, 
            'reg_alpha': 3.0397831068256878, 
            'reg_lambda': 8.303780324301862,
            'random_state': 42} 

        # Final Model Fit
        print("4. Fitting model...")
        pipe = build_pipeline(preprocessor, params)
        pipe.fit(X_train, y_train) 

        joblib.dump(pipe, model_path)
        print(f"\t[💾] Model berhasil diekspor ke: {model_path}\n")

        # Evaluate
        print(f"5. [📄] Finalizing classification report...")
        acc, report = evaluate(pipe, X_test, y_test)
        
        y_pred_proba = pipe.predict_proba(X_test) # HITUNG AUC MANUAL (Pengganti fungsi dari tuning.py)
        if len(numpy.unique(y)) == 2:
            # Binary Classification
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            # Multiclass
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        f1_weighted = report["weighted avg"]["f1-score"]
        precision_weighted = report["weighted avg"]["precision"]
        recall_weighted = report["weighted avg"]["recall"]
        f1_macro = report["macro avg"]["f1-score"]
        

        # Extract & Logging 
        # Kita log params kosong (atau bisa dihapus baris ini jika tidak mau log {})
        mlflow.log_params(params) 

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("precision_weighted", precision_weighted)
        mlflow.log_metric("recall_weighted", recall_weighted)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.sklearn.log_model(pipe, "model")

        print(f"""
            
    6. [📜] Report Evaluation:
        Accuracy\t\t: {acc},
        f1_weighted\t\t: {f1_weighted},
        f1_macro\t\t: {f1_macro},
        recall\t\t: {recall_weighted},
        precision\t\t: {precision_weighted},
        roc_auc\t\t: {auc}\n
    """)

        print(f"[✅] Runtime has been saved! Terminal: `mlflow ui`")
        new_run_id = run.info.run_id
        print(f"[✅] Run complete! New ID: {new_run_id}")
            
        mlflow.end_run()


if __name__ == '__main__':
    main()





