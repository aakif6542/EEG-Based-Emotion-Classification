# 🧠 NeuroAI Pipeline — EEG Emotion Recognition

A machine learning pipeline that reads brainwave (EEG) signals and predicts what emotion a person is feeling. It uses the **DEAP dataset** and a **Random Forest** classifier, with a simple **Streamlit** web dashboard to visualize results.

---

## What Does This Project Do?

1. **Reads EEG brain data** from the DEAP dataset (preprocessed `.dat` files).
2. **Cleans the signals** — applies a bandpass filter (0.5–40 Hz) and normalizes them.
3. **Extracts useful features** from the signals — things like average activity, variance, and power in different brain frequency bands (Delta, Theta, Alpha, Beta).
4. **Trains a Random Forest model** to classify emotions based on those features.
5. **Predicts emotions in real-time style** — slides a window across a trial and predicts the emotion at each step.
6. **Smooths the predictions** using majority voting so the output doesn't jump around.
7. **Shows everything on a web dashboard** — final emotion, emotion timeline, and confidence over time.

### Emotion Classes

Emotions are mapped from the DEAP dataset's **valence** (happy ↔ sad) and **arousal** (calm ↔ excited) scores:

| Valence | Arousal | Emotion    |
|---------|---------|------------|
| High    | High    | Excited    |
| High    | Low     | Calm       |
| Low     | High    | Stressed   |
| Low     | Low     | Sad        |

Ambiguous samples (scores between 4–6) are thrown out to keep things clean.

---

## Project Structure

```
NeuroAI Pipeline/
├── NeuroAI.ipynb          # Full pipeline — preprocessing, training, evaluation
├── app.py                 # Streamlit web app for real-time demo
├── emotion_model.pkl      # Trained Random Forest model
├── label_encoder.pkl      # Label encoder (maps numbers to emotion names)
├── requirements.txt       # Python dependencies
└── README.md              # You're reading this
```

---

## How to Set It Up

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the DEAP Dataset

- Download the **preprocessed Python** version of the DEAP dataset from [the official site](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
- Place the `.dat` files in a folder on your machine.
- Update the file path in `NeuroAI.ipynb` and `app.py` to point to your data location.

### 3. Train the Model (Optional — a pre-trained model is already included)

Open `NeuroAI.ipynb` in Jupyter and run all cells. This will:
- Load and preprocess the EEG data
- Extract features and train the model
- Save `emotion_model.pkl` and `label_encoder.pkl`

### 4. Run the Dashboard

```bash
streamlit run app.py
```

This will open a browser tab showing:
- 🎯 The **final predicted emotion**
- 📊 An **emotion timeline** chart
- 📈 A **confidence score** chart over time

---

## Tech Stack

- **Python 3.10+**
- **NumPy / SciPy** — signal processing and feature extraction
- **scikit-learn** — Random Forest classifier
- **MNE** — EEG utilities (imported in notebook)
- **Streamlit** — web dashboard
- **joblib** — model serialization

---

## Current Results

Using subject `s01` from the DEAP dataset:

| Metric     | Value  |
|------------|--------|
| Accuracy   | ~65%   |
| Classes    | Calm, Sad, Stressed |
| Features   | 280 per window |

> **Note:** The model currently trains on only one subject. Accuracy and class balance will vary across subjects.

---

## Known Limitations

- Only uses data from **one subject (s01)** — not generalized across people.
- The **"Calm" class has very low recall** (~3%) because there aren't enough Calm samples in this subject's data.
- Data file paths are **hardcoded** — you'll need to update them to match your system.
- The app does **simulated** real-time prediction (processes a full trial at once), not actual live EEG streaming.

---

## License

This project is for educational and research purposes.
