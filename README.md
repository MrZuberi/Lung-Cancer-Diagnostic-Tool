# **Lung Cancer Survival Prediction**</h6>
## ***[Demo](https://youtu.be/s_uRpYBzmaA)***

**This project focuses on building a machine learning model to predict patient survival from lung cancer using a clinical dataset. The script details the entire workflow, from data cleaning and feature engineering to model training, evaluation, and saving the final model.**

*This tool is suitable for machine learning modeling, analysis, and educational projects — not for clinical or diagnostic use.
____________________

## 📊 Dataset
The data is sourced from a synthetic lung cancer patient dataset created by Khwaish Saxena. It contains factitious patient data, including demographics, medical history, and treatment details from various countries. The dataset has no missing values, providing a clean foundation for data analysis.


## ⚙️ Feature Engineering
Before modeling, the raw data was preprocessed to create meaningful features for the classifier:


Treatment Duration: A new feature, treatment_duration_days, was created by calculating the difference between the end_treatment_date and diagnosis_date. The original date columns were then removed.
Categorical Encoding: One-hot encoding was applied to nominal categorical features such as gender, country, family_history, smoking_status, and treatment_type.
Ordinal Encoding: The cancer_stage column was converted from text (e.g., 'Stage I') to ordered numerical codes (0, 1, 2, 3) to preserve its inherent order.


## 🧠 Model Development
The goal was to build a reliable binary classification model.
An initial XGBoost Classifier was trained but did not yield satisfactory performance. Consequently, a Logistic Regression model was implemented and selected as the final model due to its robust and interpretable results.

The final model's performance on the test set is as follows:
* Accuracy: ~0.22
* Precision: ~0.41
* Recall: ~0.56
* AUC: 0.59

# ***Installation***

**Install dependencies**
```bash
pip install -r requirements.txt
```
Includes the following libraries:

-----------------------------------------------------

* pandas
* numpy 
* scikit-learn
* xgboost
* matplotlib
* joblib
* kagglehub
* streamlit

Clone the repository:

```bash
git clone https://github.com/MrZuberi/lung-cancer-prediction.git
```


## 🚀 How to Use

**Refer To READ.MD**

 # ***Our Team***
 
 Faba Kouyate
 
 Ubaid Ur Rehman
 
 Asiya Farooqi

 Taha Zuberi
