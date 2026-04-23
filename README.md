# EEG Emotion Recognition using DEAP Dataset

This project explores classification of emotional states from EEG signals using frequency-domain features and the Valence-Arousal model.

## Methods
- Bandpass filtering and normalization  
- Feature extraction: mean, variance, and band power (Delta, Theta, Alpha, Beta)  
- Power Spectral Density (PSD) for frequency analysis  
- Random Forest classifier  
- Sliding window + majority voting for temporal stabilization  

## Dataset
- DEAP EEG dataset  
- Single-subject (s01)

## Results
- Achieved stable classification across valence-arousal states  
- Improved temporal consistency using smoothing techniques  

## Limitations
- Single-subject model (limited generalization)  
- Class imbalance affecting certain emotional states  
- Simulated real-time inference  

## Conclusion
EEG-based emotion recognition is feasible using frequency-domain features, but generalization across subjects remains a key challenge.
