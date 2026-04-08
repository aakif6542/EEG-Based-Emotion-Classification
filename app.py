import streamlit as st
import numpy as np
import pandas as pd
import joblib
import scipy.signal as signal
from collections import Counter
import os
import altair as alt

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("emotion_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------
def time_domain_features(eeg_window):
    features = []
    for ch in eeg_window:
        features.extend([np.mean(ch), np.std(ch), np.var(ch)])
    return features

def bandpower(data, fs, band):
    freqs, psd = signal.welch(data, fs=fs)
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return np.sum(psd[idx])

def frequency_features(eeg_window, fs=128):
    features = []
    bands = [(0.5,4), (4,8), (8,13), (13,30)]
    
    for ch in eeg_window:
        for band in bands:
            features.append(bandpower(ch, fs, band))
    return features

def extract_features(eeg_window):
    return np.array(
        time_domain_features(eeg_window) +
        frequency_features(eeg_window)
    )

# -----------------------------
# STREAMLIT UI HEADER
# -----------------------------
st.title("🧠 EEG Emotion Recognition")
st.write("Simulated Real-Time Prediction")

# -----------------------------
# LOAD SAMPLE EEG
# -----------------------------
DATA_PATH = r"E:\Downloads\archive (6)\deap-dataset\data_preprocessed_python\s01.dat"

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find {DATA_PATH}. Please check if the file is extracted at this location.")
    st.stop()

@st.cache_data
def load_data(file_path):
    data = np.load(file_path, allow_pickle=True, encoding='latin1')
    return data['data']

eeg = load_data(DATA_PATH)
trial = np.array(eeg[0])

# -----------------------------
# STREAMLIT UI PREDICTION
# -----------------------------
window_size = 256
step = 128

predictions = []
confidence_scores = []

start = 0

while start + window_size <= trial.shape[1]:
    
    window = trial[:, start:start + window_size]
    
    features = extract_features(window).reshape(1, -1)
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    emotion = le.inverse_transform([pred])[0]
    
    predictions.append(emotion)
    confidence_scores.append(float(np.max(prob)))
    
    start += step

# -----------------------------
# SMOOTHING
# -----------------------------
def smooth_predictions(preds, window=5):
    smoothed = []
    for i in range(len(preds)):
        start_idx = max(0, i-window)
        chunk = preds[start_idx:i+1]
        
        if chunk:
            smoothed.append(Counter(chunk).most_common(1)[0][0])
    return smoothed

smooth_preds = smooth_predictions(predictions)

final_emotion = Counter(smooth_preds).most_common(1)[0][0]

# -----------------------------
# DISPLAY
# -----------------------------
st.subheader("🎯 Final Emotion")
st.success(final_emotion)

st.subheader("📊 Emotion Timeline")
df_timeline = pd.DataFrame({
    "Time Window": range(len(smooth_preds)),
    "Emotion": smooth_preds
})

timeline_chart = alt.Chart(df_timeline).mark_line(point=True).encode(
    x=alt.X("Time Window:Q", title="Time Window (Index)"),
    y=alt.Y("Emotion:N", title="Predicted Emotion")
).properties(height=350)
st.altair_chart(timeline_chart, use_container_width=True)

st.subheader("📈 Confidence Over Time")
df_conf = pd.DataFrame({
    "Time Window": range(len(confidence_scores)),
    "Confidence": confidence_scores
})

conf_chart = alt.Chart(df_conf).mark_line(point=True).encode(
    x=alt.X("Time Window:Q", title="Time Window (Index)"),
    y=alt.Y("Confidence:Q", title="Confidence Score", scale=alt.Scale(domain=[0, 1]))
).properties(height=300)
st.altair_chart(conf_chart, use_container_width=True)