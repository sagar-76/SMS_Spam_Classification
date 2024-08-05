# see i will make simple model jo msge lega in text or return krdega
# spam hai ya ni
# so for text input go to streamlit documentation and copy he text input
# import streamlit as st
# import pickle
# import string
# import nltk
# from nltk.corpus import  stopwords
# from nltk.stem.porter import PorterStemmer
#
# ps=PorterStemmer()
# tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
# model=pickle.load(open('model.pkl', 'rb'))
#
# st.title('Email/SMS  Spam Classifier')
#
# input_sms=st.text_input('Enter the message')
# # 3 steps
# # 1 text preprocessing, 2 Vectorize, 3 predict, 4 display
# # for preprocessing
#
#
#
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
# if st.button("Predict"):
#     transformed_sms = transform_text(input_sms)
#     # 2 vectorize it
#     vector_input = tfidf.transform([transformed_sms])
#     # 3 predict
#     result = model.predict(vector_input)[0]
#     if result == 1:
#         st.header('Spam')
#     else:
#         st.head('Not Spam')
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')


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

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    # Fit the vectorizer with your training data
    # Assuming you have your training data saved in some variable 'X_train'
    # Vectorize the transformed_sms
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
