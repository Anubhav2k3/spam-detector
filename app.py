import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Spam Detector')

input_msg = st.text_input("Enter the message to check for spam")

if input_msg:
    transformed_sms = transform_text(input_msg)

    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Displaying the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
