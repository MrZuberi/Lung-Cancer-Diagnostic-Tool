***Lung Cancer Survival Prediction***
This project focuses on building a machine learning model to predict patient survival from lung cancer using a clinical dataset. The script details the entire workflow, from data cleaning and feature engineering to model training, evaluation, and saving the final model.

📊 ***Dataset***
The data is sourced from the Lung Cancer Dataset on Kaggle. It contains anonymized patient data, including demographics, medical history, and treatment details from various, mostly European, countries. The dataset has no missing values, providing a clean foundation for analysis.

⚙️ ***Feature Engineering***
Before modeling, the raw data was preprocessed to create meaningful features for the classifier:

Treatment Duration: A new feature, treatment_duration_days, was created by calculating the difference between the end_treatment_date and diagnosis_date. The original date columns were then removed.
Categorical Encoding: One-hot encoding was applied to nominal categorical features such as gender, country, family_history, smoking_status, and treatment_type.
Ordinal Encoding: The cancer_stage column was converted from text (e.g., 'Stage I') to ordered numerical codes (0, 1, 2, 3) to preserve its inherent order.

🧠 ***Model Development***
The goal was to build a reliable binary classification model.
An initial XGBoost Classifier was trained but did not yield satisfactory performance. Consequently, a Logistic Regression model was implemented and selected as the final model due to its robust and interpretable results.

The final model's performance on the test set is as follows:
Accuracy: ~0.77
Precision: ~0.73
Recall: ~0.61
AUC: 0.81

🚀 ***How to Use***
To replicate this project, follow the steps below.

Prerequisites
Ensure you have Python installed, along with the following libraries:

pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
kagglehub

Installation
Clone the repository:

Bash
git clone https://github.com/your-username/lung-cancer-prediction.git
cd lung-cancer-prediction

***Install dependencies:***

Bash
pip install -r requirements.txt
Running the Script
Execute the main Python script to run the full data processing and model training pipeline.

Bash
python lung_cancer___0_77_accuracy.py
