import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
model_rf = pickle.load(open(os.path.join(BASE_DIR, "model_rf.pkl"), "rb"))
model_svm = pickle.load(open(os.path.join(BASE_DIR, "model_svm.pkl"), "rb"))
model_log = pickle.load(open(os.path.join(BASE_DIR, "model_log.pkl"), "rb"))
model_voting = pickle.load(open(os.path.join(BASE_DIR, "model_voting.pkl"), "rb"))
df_cols = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))


df_cols = pickle.load(open("columns.pkl", "rb"))

st.title("Heart Disease Prediction")

model_choice = st.selectbox(
    "Pilih Model",
    ("VotingClassifier", "RandomForest", "SVM", "LogisticRegression")
)

input_data = {}

st.subheader("Masukkan Data Input")

for col in df_cols:
    val = st.number_input(col, value=0.0)
    input_data[col] = val

def predict(model_name, data):
    arr = np.array(list(data.values()), dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    if model_name == "RandomForest":
        model = model_rf
    elif model_name == "SVM":
        model = model_svm
    elif model_name == "LogisticRegression":
        model = model_log
    else:
        model = model_voting

    pred = model.predict(arr_scaled)[0]
    try:
        prob = model.predict_proba(arr_scaled)[0].max()
    except:
        prob = None

    label = "Punya penyakit jantung" if pred == 1 else "Tidak punya penyakit jantung"

    return label, prob

if st.button("Prediksi"):
    label, prob = predict(model_choice, input_data)
    st.subheader("Hasil Prediksi:")
    st.write(label)
    if prob is not None:
        st.write("Confidence:", round(prob * 100, 2), "%")
