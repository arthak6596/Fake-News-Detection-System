import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    return PorterStemmer()

ps = download_nltk_resources()

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, vectorizer = load_model()

def preprocess_text(text):
    """Preprocess the input text similar to how training data was processed"""
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    # Tokenize
    text = text.split()
    
    # Remove stopwords and stem
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    
    # Join back to string
    return ' '.join(text)

# Create the Streamlit app
st.title('Fake News Detector')
st.write('Enter a news article to check if it might be fake or real.')

# Text input
news_text = st.text_area('News text:', height=200)

# Prediction button
if st.button('Check Authenticity'):
    if news_text and model is not None and vectorizer is not None:
        try:
            # Preprocess text
            processed_text = preprocess_text(news_text)
            
            # Vectorize - fix for the error
            text_vector = vectorizer.transform([processed_text])
            
            # Predict
            prediction = model.predict(text_vector)[0]
            
            # Display result
            if prediction == 1:
                st.error('This appears to be FAKE news.')
            else:
                st.success('This appears to be REAL news.')
            
            # Show confidence if the model supports it
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_vector)[0]
                confidence = max(probabilities) * 100
                st.write(f'Confidence: {confidence:.2f}%')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Try checking if your model and vectorizer are compatible with your current scikit-learn version")
    elif model is None or vectorizer is None:
        st.error("Failed to load model or vectorizer. Please check the files and paths.")
    else:
        st.warning('Please enter some text to analyze.')