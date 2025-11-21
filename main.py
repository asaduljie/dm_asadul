import pickle
import numpy as np

scaler = pickle.load(open("scaler.pkl", "rb"))
model_rf = pickle.load(open("model_rf.pkl", "rb"))
model_svm = pickle.load(open("model_svm.pkl", "rb"))
model_log = pickle.load(open("model_log.pkl", "rb"))
model_voting = pickle.load(open("model_voting.pkl", "rb"))

def predict(model_name, input_dict):
    features = np.array(list(input_dict.values()), dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features)

    if model_name == "rf":
        model = model_rf
    elif model_name == "svm":
        model = model_svm
    elif model_name == "log":
        model = model_log
    else:
        model = model_voting

    pred = model.predict(features_scaled)[0]

    try:
        prob = model.predict_proba(features_scaled)[0].max()
    except:
        prob = None

    result = "Punya penyakit jantung" if pred == 1 else "Tidak punya penyakit jantung"

    return {
        "prediction": int(pred),
        "label": result,
        "confidence": float(prob) if prob is not None else None
    }

if __name__ == "__main__":
    example_input = {
        "age": 60,
        "trestbps": 140,
        "chol": 230,
        "thalch": 150,
        "oldpeak": 2.3,
        "ca": 0,
        "sex_Male": 1,
        "dataset_Hungary": 0,
        "dataset_Switzerland": 0,
        "dataset_VA Long Beach": 0,
        "cp_atypical angina": 0,
        "cp_non-anginal": 1,
        "cp_typical angina": 0,
        "fbs_True": 0,
        "restecg_normal": 1,
        "restecg_st-t abnormality": 0,
        "exang_True": 1,
        "slope_flat": 1,
        "slope_upsloping": 0,
        "thal_normal": 1,
        "thal_reversable defect": 0
    }

    output = predict("voting", example_input)
    print(output)
