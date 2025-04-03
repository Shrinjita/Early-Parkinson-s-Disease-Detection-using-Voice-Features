import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load saved model and scaler
model = joblib.load('parkinsons_ensemble_model.pkl')
scaler = joblib.load('parkinsons_scaler.pkl')

# Feature names
features = ['MDVP:Jitter(%)', 'MDVP:Shimmer(dB)', 'HNR', 'PPE', 
            'spread1', 'D2', 'RPDE', 'DFA', 'spread2', 'DDA']

st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")

st.title("Early Parkinson's Disease Detection using Voice Analysis")
st.markdown("""
This app predicts the likelihood of Parkinson's disease based on vocal features.
The model achieves 89% precision using state-of-the-art machine learning techniques.
""")

# Input form
with st.expander("Enter Voice Feature Measurements"):
    input_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        input_data[feature] = cols[i%2].number_input(feature, step=0.0001, format="%.4f")

# Prediction and results
if st.button('Predict Parkinson\'s Risk'):
    # Prepare input
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]
    
    # Display results
    st.subheader("Results")
    if prediction[0] == 1:
        st.error(f"High risk of Parkinson's disease ({(probability*100):.1f}% probability)")
    else:
        st.success(f"Low risk of Parkinson's disease ({(1-probability)*100:.1f}% probability)")
    
    # Probability gauge
    fig = px.bar(x=[probability], y=['Risk Score'], orientation='h',
                 range_x=[0,1], text=np.round([probability], 2),
                 color_discrete_sequence=['red' if prediction[0]==1 else 'green'])
    fig.update_layout(showlegend=False, xaxis_title="Probability",
                      yaxis_title="", width=700, height=200)
    st.plotly_chart(fig)
    
    # Feature importance visualization
    st.subheader("Key Contributing Features")
    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values = explainer.shap_values(scaled_input)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, scaled_input, feature_names=features, 
                      plot_type="bar", show=False)
    st.pyplot(fig)

    # Instructions
with st.expander("How to use this app"):
    st.markdown("""
    1. Enter voice feature measurements from acoustic analysis
    2. Click 'Predict Parkinson's Risk' button
    3. View the prediction results and probability score
    4. For clinical use, consult with a medical professional
    """)