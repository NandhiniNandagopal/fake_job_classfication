import streamlit as st
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="Fake Job Offer Detector",
    page_icon="🚨",
    layout="centered"
)

# ===============================
# Aesthetic Theme
# ===============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
h1 {
    text-align: center;
    color: #ffffff;
}
textarea {
    background-color: #1f2a40 !important;
    color: white !important;
}
.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Load Model Files
# ===============================

model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ===============================
# UI
# ===============================

st.title("🚨 Fake Job Offer Detection System")
st.write("Paste a job offer paragraph below to verify its authenticity.")

job_text = st.text_area("Enter Job Offer Text Here")

# ===============================
# Prediction
# ===============================

if st.button("🔍 Analyze Offer"):

    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Transform text using TF-IDF
        text_features = vectorizer.transform([job_text])

        # Create zero padding for other features (if model expects them)
        dummy_other_features = csr_matrix((1, model.coef_.shape[1] - text_features.shape[1]))

        final_input = hstack([text_features, dummy_other_features])

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

        if prediction == 1:
            st.error("It's a fake offer, don’t get fooled.")
        else:
            st.success("It's a real offer, you can trust this message.")
