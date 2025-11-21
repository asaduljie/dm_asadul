import streamlit as st
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(filename):
    for root, dirs, files in os.walk(BASE_DIR):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(filename)

scaler = pickle.load(open(load_file("scaler.pkl"), "rb"))
model_rf = pickle.load(open(load_file("model_rf.pkl"), "rb"))
model_svm = pickle.load(open(load_file("model_svm.pkl"), "rb"))
model_log = pickle.load(open(load_file("model_log.pkl"), "rb"))
model_voting = pickle.load(open(load_file("model_voting.pkl"), "rb"))
columns = pickle.load(open(load_file("columns.pkl"), "rb"))

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

st.markdown("""
    <h1 style="text-align:center; font-size:40px;">
        ❤️ Sistem Prediksi Penyakit Jantung
    </h1>
    <p style="text-align:center; font-size:18px; opacity:0.8;">
        Silakan masukkan data pasien secara lengkap untuk memprediksi risiko penyakit jantung.
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia Pasien", 20, 100, 50)
    trestbps = st.slider("Tekanan Darah Saat Istirahat (mmHg)", 80, 200, 120)
    chol = st.slider("Kadar Kolesterol (mg/dl)", 100, 400, 200)
    thalach = st.slider("Detak Jantung Maksimal (BPM)", 50, 210, 150)
    oldpeak = st.slider("Depresi ST (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    ca = st.slider("Jumlah Pembuluh Darah Besar (0–4)", 0, 4, 0)

with col2:
    sex = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    dataset = st.radio("Sumber Data Pasien", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
    cp = st.selectbox("Tipe Nyeri Dada", [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ])
    fbs = st.radio("Gula Darah Puasa > 120 mg/dl?", ["Ya", "Tidak"])
    restecg = st.selectbox("Hasil Pemeriksaan ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
    exang = st.radio("Angina Saat Aktivitas Fisik?", ["Ya", "Tidak"])
    slope = st.selectbox("Kemiringan Segmen ST", ["upsloping", "flat", "downsloping"])
    thal = st.selectbox("Kondisi Thalium Stress Test", ["normal", "fixed defect", "reversible defect"])

category_data = {
    "sex": sex,
    "dataset": dataset,
    "cp": cp,
    "fbs": fbs,
    "restecg": restecg,
    "exang": exang,
    "slope": slope,
    "thal": thal
}

input_final = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "thalch": thalach,
    "oldpeak": oldpeak,
    "ca": ca
}

dummy_template = {col: 0 for col in columns if col not in input_final}

mapping = {
    ("sex", "Laki-laki"): "sex_Male",
    ("dataset", "Hungary"): "dataset_Hungary",
    ("dataset", "Switzerland"): "dataset_Switzerland",
    ("dataset", "VA Long Beach"): "dataset_VA Long Beach",
    ("cp", "atypical angina"): "cp_atypical angina",
    ("cp", "non-anginal"): "cp_non-anginal",
    ("cp", "typical angina"): "cp_typical angina",
    ("fbs", "Ya"): "fbs_True",
    ("restecg", "normal"): "restecg_normal",
    ("restecg", "st-t abnormality"): "restecg_st-t abnormality",
    ("exang", "Ya"): "exang_True",
    ("slope", "flat"): "slope_flat",
    ("slope", "upsloping"): "slope_upsloping",
    ("thal", "normal"): "thal_normal",
    ("thal", "reversible defect"): "thal_reversable defect"
}

for (feature, value), col_name in mapping.items():
    if category_data[feature] == value:
        dummy_template[col_name] = 1

final_row = np.array(list(input_final.values()) + list(dummy_template.values())).reshape(1, -1)
final_scaled = scaler.transform(final_row)

model_choice = st.selectbox("Pilih Model Prediksi", [
    "VotingClassifier",
    "RandomForest",
    "SVM",
    "Logistic Regression"
])

if st.button("🔍 Prediksi Sekarang"):
    if model_choice == "RandomForest":
        model = model_rf
    elif model_choice == "SVM":
        model = model_svm
    elif model_choice == "Logistic Regression":
        model = model_log
    else:
        model = model_voting

    pred = model.predict(final_scaled)[0]
    prob = model.predict_proba(final_scaled)[0].max()

    result = "💔 Pasien Berisiko Mengalami Penyakit Jantung" if pred == 1 else "❤️ Pasien Tidak Berisiko Penyakit Jantung"

    st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background-color:#1E1E1E; text-align:center; margin-top:20px;">
            <h2 style="color:white;">{result}</h2>
            <p style="color:#ccc; font-size:18px;">Tingkat Keyakinan Prediksi: <b>{prob*100:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)
