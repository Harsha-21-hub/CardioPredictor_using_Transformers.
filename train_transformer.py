import pandas as pd
import numpy as np
import torch     # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Load & Clean Data ---
print("Loading cardio_train.csv...")
df = pd.read_csv('cardio_train.csv', sep=';')

# Drop ID (not predictive)
df.drop('id', axis=1, inplace=True)

# Cleaning: Filter out impossible Blood Pressure values (Outliers)
# Systolic (ap_hi) should be roughly 60-240, Diastolic (ap_lo) 40-160
df = df[(df['ap_hi'] > 60) & (df['ap_hi'] < 240)]
df = df[(df['ap_lo'] > 40) & (df['ap_lo'] < 160)]

X = df.drop("cardio", axis=1).values
y = df["cardio"].values

# Scale Data (Vital for Neural Networks)
# Features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- 2. Define the Tabular Transformer ---
class CardioTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(CardioTransformer, self).__init__()
        
        # Project each feature into a higher dimensional space
        self.embedding = nn.Linear(1, d_model) 
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (Batch, Features) -> (Batch, Features, 1)
        x = x.unsqueeze(-1)
        
        # Embed -> (Batch, Features, 64)
        x = self.embedding(x)
        
        # Attention Mechanism
        x = self.transformer_encoder(x)
        
        # Flatten -> (Batch, Features*64)
        x = x.view(x.size(0), -1)
        
        # Output Probability
        output = self.fc(x)
        return output

# --- 3. Train ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = X.shape[1] # Should be 11
model = CardioTransformer(num_features=num_features).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on {len(X_train)} samples...")
epochs = 20 
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

# --- 4. Evaluate ---
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    predicted = (test_outputs > 0.5).float()
    acc = accuracy_score(y_test, predicted.cpu().numpy())

print(f"✅ Final Accuracy: {acc:.2%}")

# --- 5. Save ---
torch.save(model.state_dict(), 'cardio_transformer.pth')
joblib.dump(scaler, 'scaler_custom.pkl')
print("Model saved as 'cardio_transformer.pth'")