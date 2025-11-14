import streamlit as st
import pickle
import numpy as np
import re
import string

@st.cache_resource
def load_artifacts():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    embedding_matrix = np.load('embedding_matrix.npy')
    
    return model, vectorizer, embedding_matrix

model, vectorizer, embedding_matrix = load_artifacts()

def preprocess_text(sen):
  sen = sen.lower()
  sen = sen.translate(str.maketrans('', '', string.punctuation))
  sen = re.sub('[^a-zA-Z]', ' ', sen)
  sen = re.sub(r'\s+[a-zA-Z]\s+', ' ', sen)
  sen = re.sub(r'\s+', ' ', sen)
  return sen


st.title('Toxic Comment Classifier')
st.write('Enter a comment below to see if it\'s classified as toxic or not.')
user_input = st.text_area('Comment:', 'Enter your text here...')

if st.button('Classify'):
    if user_input:
        processed_input = preprocess_text(user_input)
        tfidf_vec = vectorizer.transform([processed_input])
        doc_embed = tfidf_vec.dot(embedding_matrix)
        prediction = model.predict(doc_embed)[0] # Will be True or False
        probability = model.predict_proba(doc_embed)[0][1] # Prob of "Toxic"
        if prediction == True:
            st.error(f'**Classification: TOXIC** (Probability: {probability:.2%})')
        else:
            st.success(f'**Classification: NOT TOXIC** (Probability: {1-probability:.2%})')
            
    else:
        st.warning('Please enter a comment to classify.')