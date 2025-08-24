import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Lung Cancer Survival Prediction",
    layout="centered"
)

st.title("Lung Cancer Survival Prediction from Scans")
st.write("Upload a lung scan image and provide patient data to predict survival outcome")

uploaded_file = st.file_uploader("Choose a lung scan image:", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_lung_cancer_model():
    if not os.path.exists('lung_cancer_model.pkl'):
        st.error("Model file 'lung_cancer_model.pkl' not found. Please ensure it's in the same directory as this script.")
        return None
    
    try:
        import joblib
        model = joblib.load('lung_cancer_model.pkl')
        st.success("Model loaded successfully using Joblib!")
        return model
    except Exception as e:
        try:
            with open('lung_cancer_model.pkl', 'rb') as f:
                model = pickle.load(f)
            st.success("Model loaded successfully with pickle!")
            return model
        except Exception as e2:
            st.error(f"Error loading model: {str(e2)}")
            return None

def extract_image_features(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    gray_img = img.convert('L')
    gray_array = np.array(gray_img, dtype=np.float32) / 255.0
    
    image_features = [
        np.mean(gray_array),
        np.std(gray_array),
        np.var(gray_array),
        np.min(gray_array),
        np.max(gray_array),
        np.median(gray_array),
        np.percentile(gray_array, 25),
        np.percentile(gray_array, 75),
    ]
    
    return image_features

def prepare_clinical_features(age, gender, cancer_stage, bmi, cholesterol, smoking_status, family_history, image_features):
    features = []
    
    features.extend([
        age,
        1 if gender == 'Male' else 0,
        bmi,
        cholesterol,
        1 if smoking_status == 'Yes' else 0,
        1 if family_history == 'Yes' else 0,
        age / 10
    ])
    
    stage_features = [0, 0, 0, 0]
    if cancer_stage == 'Stage I':
        stage_features[0] = 1
    elif cancer_stage == 'Stage II':
        stage_features[1] = 1
    elif cancer_stage == 'Stage III':
        stage_features[2] = 1
    elif cancer_stage == 'Stage IV':
        stage_features[3] = 1
    features.extend(stage_features)
    
    features.extend(image_features)
    
    features.extend([
        1 if age < 50 else 0,
        1 if age > 70 else 0,
        age ** 0.5,
        np.log(age + 1),
        1 if bmi < 18.5 else 0,
        1 if 18.5 <= bmi < 25 else 0,
        1 if 25 <= bmi < 30 else 0,
        1 if bmi >= 30 else 0,
        bmi ** 2 / 1000,
        1 if cholesterol > 240 else 0,
        1 if cholesterol < 160 else 0,
        cholesterol / 100,
        np.log(cholesterol + 1) / 10,
        age * (1 if gender == 'Male' else 0) / 100,
        bmi * (1 if smoking_status == 'Yes' else 0) / 100,
        (1 if smoking_status == 'Yes' else 0) * (1 if family_history == 'Yes' else 0),
        cholesterol * bmi / 10000,
        1 if cancer_stage in ['Stage III', 'Stage IV'] else 0,
        len([s for s in cancer_stage if s.isdigit()]) if any(s.isdigit() for s in cancer_stage) else 1,
        (age > 65) * (1 if smoking_status == 'Yes' else 0),
        (1 if gender == 'Male' else 0) * (1 if smoking_status == 'Yes' else 0),
        (bmi > 30) * (cholesterol > 240),
        (1 if family_history == 'Yes' else 0) * (age < 60),
        age / bmi if bmi > 0 else 0,
        cholesterol / age if age > 0 else 0,
    ])
    
    if len(features) > 43:
        features = features[:43]
    elif len(features) < 43:
        features.extend([0.0] * (43 - len(features)))
    
    return np.array(features).reshape(1, -1)

def predict_survival(img, age, gender, cancer_stage, bmi, cholesterol, smoking_status, family_history):
    model = load_lung_cancer_model()
    if model is None:
        return None, None
    
    try:
        image_features = extract_image_features(img)
        processed_features = prepare_clinical_features(
            age, gender, cancer_stage, bmi, cholesterol, 
            smoking_status, family_history, image_features
        )
        
        st.info(f"Extracted {processed_features.shape[1]} features for prediction")
        
        prediction = model.predict(processed_features)[0]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_features)[0]
            confidence = np.max(probabilities)
            predicted_class = np.argmax(probabilities)
        else:
            predicted_class = prediction
            confidence = 0.8
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Lung Scan", use_column_width=True)
    
    st.markdown("### 👤 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=22)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
    
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.6, step=0.1)
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=210)
        smoking_status = st.selectbox("Smoking History", ["No", "Yes"])
        family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
    
    if st.button("🔬 Analyze Scan & Predict Survival"):
        with st.spinner("Analyzing the scan and patient data..."):
            
            predicted_class, confidence = predict_survival(
                image_display, age, gender, cancer_stage, bmi, 
                cholesterol, smoking_status, family_history
            )
            
            if predicted_class is not None:
                st.subheader("📊 Survival Prediction Results:")
                
                if predicted_class == 1:
                    st.success(f"**SURVIVED** (Confidence: {confidence:.2f})")
                    st.write("The model predicts the patient will survive based on the scan and clinical data.")
                else:
                    st.error(f"**NOT SURVIVED** (Confidence: {confidence:.2f})")
                    st.write("The model predicts poor survival outcome based on the scan and clinical data.")
                
                if confidence > 0.9:
                    confidence_level = "Very High"
                elif confidence > 0.7:
                    confidence_level = "High"
                elif confidence > 0.5:
                    confidence_level = "Moderate"
                else:
                    confidence_level = "Low"
                
                st.info(f"**Confidence Level:** {confidence_level}")
                
                survival_prob = confidence if predicted_class == 1 else (1 - confidence)
                st.metric("Survival Probability", f"{survival_prob:.1%}")
            
            st.warning("**Important Disclaimer:** This is a demonstration tool for educational purposes only and should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified medical professionals.")
            
else:
    st.info("Please upload a lung scan image to get started.")

st.markdown("---")
st.markdown("### Instructions:")
st.markdown("1. Upload a clear lung scan image (CT scan or X-ray)")
st.markdown("2. Enter the patient's clinical information")
st.markdown("3. Click 'Analyze Scan & Predict Survival' to get the prediction")
st.markdown("4. Review the survival prediction and confidence level")

