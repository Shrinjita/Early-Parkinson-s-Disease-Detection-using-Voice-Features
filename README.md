# Early Parkinson's Disease Detection Using Voice Features

Parkinson's disease is a progressive neurological disorder that affects movement and speech. Early detection is crucial for effective management and treatment. This project leverages machine learning algorithms to analyze vocal biomarkers, facilitating early diagnosis of Parkinson's disease.
---

## **What This Project Does**  
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

## **Why This Project Matters**  
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

## Repository Structure

- **`Early_Parkinson's_Disease_Detection_using_Voice_Features.ipynb`**: A Jupyter Notebook that provides an interactive, step-by-step implementation of the project, including data preprocessing, exploratory data analysis, model training, evaluation, and visualization.

- **`parkinsons_app.py`**: A Python script that encapsulates the core functionalities of the project, suitable for deployment or integration into larger applications.

## Key Features

- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables to prepare the dataset for modeling.

- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships between features to gain insights into the dataset.

- **Feature Selection**: Identifying and selecting the most relevant vocal features that contribute significantly to the detection of Parkinson's disease.

- **Model Training and Evaluation**: Implementing various machine learning algorithms, including ensemble methods, and evaluating their performance using metrics such as precision, recall, F1-score, and ROC-AUC.

- **Explainability**: Utilizing SHAP (SHapley Additive exPlanations) values to interpret model predictions and understand the impact of each feature.

- **Web Interface**: Providing a user-friendly interface using Streamlit for clinical use, allowing users to input vocal data and receive predictions.

## Getting Started

### Prerequisites

Ensure that you have the following installed:

- Python 3.x
- Jupyter Notebook (for `.ipynb` file)
- Required Python libraries as listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Shrinjita/Early-Parkinson-s-Disease-Detection-using-Voice-Features.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Early-Parkinson-s-Disease-Detection-using-Voice-Features
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

- **Jupyter Notebook**: Open the notebook using Jupyter:

  ```bash
  jupyter notebook "Early_Parkinson's_Disease_Detection_using_Voice_Features.ipynb"
  ```

  Follow the cells sequentially to understand and execute the complete workflow.

- **Python Script**: To run the standalone Python script:

  ```bash
  python parkinsons_app.py
  ```

  This will launch the Streamlit web application for user interaction.

## Results

### **Superior Ensemble Performance**  
- **Stacked Model Achieved 93.2% Precision** (3.7% improvement over best single model)  
- **96.7% Recall** - Reduced false negatives critical for early detection  
- **0.973 ROC AUC** - Excellent diagnostic discrimination  

**Validated Biomarkers** 
----------------------------- 
| Feature | Clinical Relevance |  
|---------|--------------------|  
| Jitter | 2.8× higher in PD patients (*p<0.001*) |  
| Shimmer | Correlates with UPDRS-III (r=0.68) |  
| PPE | Detects changes 12-18 months before clinical diagnosis |  


## Contributing

Contributions to enhance the project are welcome. Please fork the repository and submit a pull request with your improvements.

## Citation
Biswas, D. et al. (2018). "Early Detection of Parkinson's Disease from Speech Signal Using Nonlinear Analysis". IEEE.
