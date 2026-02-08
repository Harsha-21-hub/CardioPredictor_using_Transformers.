# 🧠 Cardio Predictor using Transformers

A deep learning–based heart disease prediction system using a **Transformer neural network** trained on tabular medical data.

## 🚀 Live Demo
(Deploy on Streamlit after upload)

## 🧠 Model Architecture
- Transformer Encoder for tabular data
- Attention mechanism over medical features
- Binary classification using Sigmoid output

## 📊 Features Used
- Age
- Gender
- Height, Weight
- Blood Pressure (ap_hi, ap_lo)
- Cholesterol, Glucose
- Smoking, Alcohol, Physical Activity

## 🛠 Tech Stack
- Python
- PyTorch
- scikit-learn
- Streamlit
- Plotly

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
python train_transformer.py
