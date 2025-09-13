# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")
# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)

#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms]).toarray()

#     # 3. add num_characters feature (same as training)
#     num_chars = len(input_sms)
#     vector_input = np.hstack((vector_input, [[num_chars]]))  # shape (1,3001)

#     # 4. predict
#     result = model.predict(vector_input)[0]

#     # 5. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")


import streamlit as st
import pickle
import string
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to clean and stem text
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


# Load vectorizer, scaler, and trained model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))   # make sure you saved scaler.pkl during training
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📧 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()

    # 3. Add extra feature (num_characters) same as training
    num_chars = len(input_sms)
    vector_input = np.hstack((vector_input, [[num_chars]]))  # shape (1, 3001)

    # 4. Scale (same as training)
    vector_input = scaler.transform(vector_input)

    # 5. Predict
    result = model.predict(vector_input)[0]

    # 6. Display
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")

