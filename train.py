#========================================= Library =========================================
import os
import logging
import warnings
import streamlit as st
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Modular directory finder
db_path = os.path.join(BASE_DIR, "mlflow.db")
os.environ['MLFLOW_TRACKING_URI'] = f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}"

# Comment berikut jika ingin debug
logging.basicConfig(level=logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

import mlflow
import mlflow.sklearn
import mlflow.data

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

import hashlib
import pandas
import numpy
from sklearn.metrics import roc_auc_score # TAMBAHAN: Import roc_auc_score karena kita tidak pakai tuning.py lagi
from data_ingestion import load_data, get_file_hash, handle_schema
from pre_processing import build_preprocessor
from pipeline import build_pipeline
from evaluation import evaluate
from sklearn.model_selection import train_test_split

#========================== Setting up Mlflow & File Directory ===============================
mlflow.set_tracking_uri(f"sqlite:///{db_path}") # Pastikan nama eksperimen aman (tanpa slash file path)
mlflow.set_experiment("Heart_Attack_Prediction") 

os.system('cls' if os.name == 'nt' else 'clear')

if mlflow.active_run():
    mlflow.end_run()

artifact_dir = "artifacts"
model_filename = "heart_attack_pipeline.pkl"
# model_path = "artifacts/heart_attack_pipeline.pkl"
model_path = os.path.join(artifact_dir,model_filename)

if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)
    # print(f"[📁] Folder '{artifact_dir}' berhasil dibuat.")
# else:
    # print(f"[📂] Folder '{artifact_dir}' sudah ada.")

# ================================= Executing MLflow Code =======================================================
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