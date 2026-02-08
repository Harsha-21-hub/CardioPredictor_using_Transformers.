import streamlit as st
import joblib
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

# --- Model Architecture (Must match training exactly) ---
class CardioTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(CardioTransformer, self).__init__()
        self.embedding = nn.Linear(1, d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

# --- Load Resources ---
@st.cache_resource
def load_model():
    scaler = joblib.load('scaler_custom.pkl')
    # We have 11 features in this specific dataset
    model = CardioTransformer(num_features=11)
    model.load_state_dict(torch.load('cardio_transformer.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

try:
    model, scaler = load_model()
except:
    st.error("Model not found. Run 'train_transformer.py' first!")
    st.stop()

# --- UI ---
st.set_page_config(page_title="Cardio Neural Net", layout="wide")
st.title("🧠 Neural Heart Disease Predictor")
st.markdown("Using **Transformer Attention** on custom dataset.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Vitals")
    
    # 1. Age (Days in dataset, Years in UI)
    age_years = st.number_input("Age (Years)", 20, 100, 50)
    age_days = age_years * 365.25 
    
    # 2. Gender (1=Female, 2=Male in this dataset)
    gender_ui = st.selectbox("Gender", ["Male", "Female"])
    gender = 2 if gender_ui == "Male" else 1
    
    # 3. Physical
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 75)
    
    # 4. Vitals
    ap_hi = st.number_input("Systolic BP (ap_hi)", 80, 240, 120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", 40, 160, 80)
    
    # 5. Labs (1: Normal, 2: Above Normal, 3: Well Above)
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
    gluc = st.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
    
    # 6. Habits
    smoke = st.checkbox("Smoker?")
    alco = st.checkbox("Alcohol Intake?")
    active = st.checkbox("Physically Active?")
    
    # Convert booleans to 0/1
    smoke = 1 if smoke else 0
    alco = 1 if alco else 0
    active = 1 if active else 0

    # Feature Array
    # Order: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    features = np.array([[age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

with col2:
    st.subheader("Prediction")
    
    if st.button("Analyze Risk", type="primary"):
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        tensor_in = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = model(tensor_in).item()
        
        risk_percent = prob * 100
        
        # Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percent,
            title = {'text': "Cardio Disease Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "#90EE90"}, # Light Green
                    {'range': [50, 100], 'color': "#FFCCCB"} # Light Red
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        if risk_percent > 50:
            st.error(f"🚨 High Probability ({risk_percent:.1f}%) of cardiovascular issues.")
        else:
            st.success(f"✅ Low Probability ({risk_percent:.1f}%) detected.")