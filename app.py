# --- Imports ---
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------------------
# Load + Preprocess Data
# ---------------------------
data = pd.read_csv("bank_additional_full_cleaned.csv")

# Drop columns that may leak information or are not useful
data = data.drop(columns=['duration', 'pdays', 'previous'], errors='ignore')

# Encode categorical features with LabelEncoder
le = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop('y', axis=1)
y = data['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Build ANN Model
# ---------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

# ---------------------------
# Prediction Function
# ---------------------------
def predict_bank(*features):
    # Scale input
    features_scaled = scaler.transform([features])
    proba = model.predict(features_scaled, verbose=0)[0][0]
    pred = 1 if proba > 0.5 else 0
    return {
        "Prediction": "✅ Customer will subscribe" if pred == 1 else "❌ Customer will not subscribe"    }

# ---------------------------
# Login Function
# ---------------------------
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login. Try again."), gr.update(visible=False)

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    # Page 1: Login
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔑 Login to Access Bank Subscription ANN App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: ANN App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🏦 Bank Subscription Prediction (ANN)")

            inputs = []
            with gr.Accordion("Enter Feature Values", open=False):
                for col in X.columns:
                    inputs.append(gr.Number(label=col, value=float(X[col].median())))

            btn = gr.Button("Predict Subscription")
            output = gr.JSON(label="Result")

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict_bank, inputs=inputs, outputs=output)

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
