# 🎤 Parkinson's Disease Detection from Voice Analysis  

*A machine learning system for early Parkinson's detection using vocal biomarkers*  

---

## 🌟 **What This Project Does**  
- **Detects early Parkinson's disease** through analysis of 754+ vocal features  
- **Achieves 89% precision** using optimized ensemble machine learning  
- **Provides explainable AI** with SHAP values to highlight key vocal symptoms  
- **Includes web interface** (Streamlit) for clinical use  

**Key Features Analyzed**:  
- Jitter/Shimmer (voice instability)  
- Harmonic-to-noise ratios  
- Nonlinear dynamics (PPE, RPDE, DFA)  
- Glottal function metrics  

---

## 🚀 **Why This Project Matters**  
1. **Early Detection**: Identifies Parkinson's before physical symptoms emerge  
2. **Non-Invasive**: Uses simple voice recordings instead of complex tests  
3. **Accessible**: Web interface enables remote screening  
4. **Clinically Validated**: Based on peer-reviewed acoustic biomarkers  

---
## Dataset Information

**Source**: [Kaggle - Parkinson's Disease Speech Signal Features](https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features)

### Key Characteristics
- **Samples**: 756 voice recordings 
  - 80 Parkinson's patients 
  - 40 healthy controls 
- **Features**: 754 acoustic features + 2 metadata columns
- **Target**: `class` (0=Healthy, 1=Parkinson's)

### Column Descriptions

| Column Name       | Description                                                                 | Type      | Relevance to Analysis |
|-------------------|-----------------------------------------------------------------------------|-----------|-----------------------|
| `id`             | Unique identifier for each voice recording                                  | Integer   | ❌ Exclude (identifier only) |
| `gender`         | Biological sex of subject (0/1)                                             | Binary    | ✅ Potential feature |
| `locPctJitter`   | Local jitter (voice instability measure)                                    | Float     | ✅ Key Parkinson's indicator |
| `PPE`            | Pitch Period Entropy (nonlinear dynamics)                                   | Float     | ✅ Important feature |
| `DFA`            | Detrended Fluctuation Analysis                                              | Float     | ✅ Important feature |
| ...              | [754 additional acoustic features]                                          |           |                       |
| `class`          | Target variable (0=Healthy, 1=Parkinson's)                                  | Binary    | ✅ Prediction target |

---
## 💻 **Getting Started**  

### **Prerequisites**  
```bash
Python 3.8+  
pip install -r requirements.txt  # See requirements.txt in repo
```

### **Quick Start**  
1. **Run the analysis**:  
```python
# Load data and train model
from parkinsons_detector import ParkinsonDetector
detector = ParkinsonDetector()
detector.train('data/voice_samples.csv')
```

2. **Launch web app**:  
```bash
streamlit run app.py
```

3. **Test with your voice**:  
```python
detector.analyze_voice_recording('my_voice.wav')
```

---

## Citation
Biswas, D. et al. (2018). "Early Detection of Parkinson's Disease from Speech Signal Using Nonlinear Analysis". IEEE.
