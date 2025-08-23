# Lung Cancer Diagnostic Tool for the NeuraVia Hackathon

Lung Cancer Diagnostic Tool (NeuraVia Hackathon)
A proof-of-concept web application that uses a machine learning model to provide an initial risk assessment for lung cancer based on patient data. This tool is designed to assist healthcare professionals in the early stages of diagnosis.

🎯 Project Goal
Early detection of lung cancer significantly increases survival rates. This project aims to provide a simple, accessible tool that leverages machine learning to quickly analyze key patient metrics and flag high-risk cases that may require further investigation.

✨ Features
User-Friendly Interface: A clean web form for easy input of patient data.

ML-Powered Prediction: The backend uses a trained neural network to generate a risk score in real-time.

Clear Results: The prediction is displayed as a clear probability percentage and risk level (e.g., Low, Medium, High).

Data Security: All data is processed in memory and is not stored, ensuring patient privacy.

🛠️ Tech Stack
Machine Learning: TensorFlow (Keras), Scikit-learn, Pandas

Backend: Python, Flask

Frontend: HTML, CSS, JavaScript

🚀 Getting Started
Follow these steps to get the application running on your local machine.

Prerequisites
Python 3.8+

Pip (Python Package Installer)
Installation & Setup
Clone the repository:

Bash
git clone https://github.com/MrZuberi/lung-cancer-diagnostic-tool.git
cd lung-cancer-diagnostic-tool

Install the required Python packages:

$ Bash
pip install -r requirements.txt

Running the Application

$ Bash
(!Exe goes here!)

📊 Model Overview
The core of this tool is a Multi-Layer Perceptron (MLP) model built with TensorFlow and Keras. It was trained on a tabular dataset containing anonymized patient data.

The primary challenge was a moderate class imbalance (78/22 split), which was addressed using class weighting during training to ensure the model did not become biased towards the majority class. This technique improved the model's ability to correctly identify the **high-risk** minority class.

🔮 Future Improvements
Model Explainability: Integrate SHAP or LIME to explain why the model made a specific prediction.

Incorporate Imaging Data: Expand the model to analyze CT scan images using a Convolutional Neural Network (CNN) for a more comprehensive diagnosis.

Cloud Deployment: Deploy the application on a cloud platform like GCP, AWS, or Azure for wider accessibility.

👥 Team Members
Faba Kouyate
Ubaid Ur Rehman
Asia (LAST NAME)
Aztakis (LAST NAME)

