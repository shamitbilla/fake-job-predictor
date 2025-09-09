import streamlit as st
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from tensorflow.keras.models import load_model

# Load saved TF-IDF Vectorizer and ANN model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = load_model('fraudulent_job_model.keras')

# Text cleaning function
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub('\[[^]]*\]', '', text)
    text = re.sub('[^a-zA-Z\s]', '', text).lower()

    stop = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop])

    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

# Streamlit App UI
st.title("🚨 Fake Job Postings Classifier")

st.write("""
Enter job post details below and the model will predict if it's **Fake or Real**.
""")

title = st.text_input('Job Title')
company_profile = st.text_area('Company Profile')
description = st.text_area('Job Description')
requirements = st.text_area('Requirements')
benefits = st.text_area('Benefits')

if st.button('Predict'):
    # Combine user input
    input_text = ' '.join([title, company_profile, description, requirements, benefits])
    clean_input = clean_text(input_text)
    
    st.write(f"🧹 Cleaned Input: '{clean_input}'")  # Debug info

    # Sanity check: Avoid empty input
    if len(clean_input.strip()) == 0:
        st.error("⚠️ Please enter valid job posting details before predicting!")
    else:
        # Vectorize and predict
        vectorized_input = tfidf_vectorizer.transform([clean_input])
        prediction_prob = model.predict(vectorized_input)[0][0]
        prediction = int(prediction_prob > 0.5)

        st.write(f"🔍 Prediction confidence: {prediction_prob:.2f}")

        if prediction == 1:
            st.error("⚠️ This job post is likely **Fake**!")
        else:
            st.success("✅ This job post looks **Real**.")
