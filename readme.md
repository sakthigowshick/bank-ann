# 🏦 Bank Subscription Prediction (ANN)

## 📌 Overview
This project builds an **Artificial Neural Network (ANN)** model to predict whether a customer will subscribe to a term deposit based on the **Bank Marketing Dataset**.  
It uses **Python** with **Pandas**, **Scikit-learn**, **TensorFlow/Keras**, and **Matplotlib** to perform preprocessing, model training, and visualization.  
An optional **Gradio app** is also included for interactive predictions.

---

## 🚀 Usage
1. Open the `bank_ann.ipynb` notebook in Jupyter Notebook or JupyterLab.  
2. Run the cells sequentially to:
   - Load and preprocess the dataset (`bank_additional_full_cleaned.csv`)
   - Encode categorical variables
   - Train the ANN model
   - Evaluate performance  
3. Run the Gradio app (`app.py`) to interactively test predictions.

---

## 📊 Results
Below are some output images generated from the analysis:

### 🔹 Training Accuracy & Loss Curve
![Training History](training_history.png)

### 🔹 Confusion Matrix of ANN on Test Data
![Confusion Matrix](confusion_matrix.png)

### 🔹 Sample Gradio App Prediction
![Gradio Demo](gradio_demo.png)

---

💡 *Feel free to fork this repo and try your own experiments with different ANN architectures or hyperparameters!*
