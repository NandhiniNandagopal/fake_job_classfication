import streamlit as st
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🚨",
    layout="centered"
)

# ===============================
# Custom CSS (Aesthetic Theme)
# ===============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: white;
}
h1 {
    text-align: center;
    color: #ffffff;
}
.stTextInput>div>div>input, 
.stTextArea textarea {
    background-color: #2b2b3c;
    color: white;
}
.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Load Model & Files
# ===============================

model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# ===============================
# UI Header
# ===============================

st.title("🚨 Fake Job Posting Detection System")
st.write("AI-powered system to detect fraudulent job offers.")

# ===============================
# User Input Fields
# ===============================

title = st.text_input("Job Title")
description = st.text_area("Job Description")
company_profile = st.text_area("Company Profile")

location = st.text_input("Location")
department = st.text_input("Department")
salary_range = st.text_input("Salary Range")
employment_type = st.text_input("Employment Type")
required_experience = st.text_input("Required Experience")
required_education = st.text_input("Required Education")
industry = st.text_input("Industry")
function = st.text_input("Function")

# ===============================
# Prediction Section
# ===============================

if st.button("🔍 Analyze Job Offer"):

    input_data = pd.DataFrame([{
        "title": title,
        "description": description,
        "company_profile": company_profile,
        "location": location,
        "department": department,
        "salary_range": salary_range,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function
    }])

    # Encode categorical columns safely
    for col in label_encoders:
        if col in input_data.columns:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except:
                input_data[col] = 0

    # Combine text
    input_data["combined_text"] = (
        input_data["title"].astype(str) + " " +
        input_data["description"].astype(str) + " " +
        input_data["company_profile"].astype(str)
    )

    # Transform text
    text_features = vectorizer.transform(input_data["combined_text"])

    other_features = input_data.drop(
        columns=["title", "description", "company_profile", "combined_text"]
    ).values

    other_features_sparse = csr_matrix(other_features)

    final_input = hstack([text_features, other_features_sparse])

    # Prediction
    prediction = model.predict(final_input)[0]
    probabilities = model.predict_proba(final_input)[0]

    real_conf = probabilities[0]
    fake_conf = probabilities[1]

    st.subheader("📊 Prediction Confidence")

    confidence_df = pd.DataFrame({
        "Class": ["Real", "Fake"],
        "Confidence": [real_conf, fake_conf]
    })

    st.bar_chart(confidence_df.set_index("Class"))

    st.markdown("---")

    # Final Message
    if prediction == 1:
        st.error("It's a fake offer, don’t get fooled.")
    else:
        st.success("It's a real offer, you can trust this message.")
