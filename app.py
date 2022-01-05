import streamlit as st
import pickle
import string
import nltk
import sklearn

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
def text_cleaner(data):                                                                     ## fuction to preprocess data
    data = data.lower()                                                                     ## lower
    data = nltk.word_tokenize(data)                                                         ## word tokenize
    new_data = []
    for i in data:
        if i.isalnum():                                                                     ## remove special character
            new_data.append(i)
    data = new_data[:]
    new_data.clear()
    for i in data:

        if i not in stopwords.words('english') and i not in string.punctuation:             ## remove stopwords and punctuation
            new_data.append(i)

    data = new_data[:]
    new_data.clear()
    for i in data:
        new_data.append(ps.stem(i))                                                          ## steming the data
    return " ".join(new_data)

model = pickle.load(open('bnb_model.pkl','rb'))
cv_vector = pickle.load(open('vector.pkl','rb'))

st.title("Depression detector")
input = st.text_area("let's share something....")
if st.button("predict"):
    preprocess_input = text_cleaner(input)
    vectorize_input = cv_vector.transform([preprocess_input])                                              ## vector of input
    result=model.predict(vectorize_input)

    if result == 1:
        st.header("Depression")
        st.subheader("“The bravest thing I ever did was continuing my life when I wanted to die.” – Juliette Lewis")
        st.image("OIP (3).jfif")


    else:
        st.header("Keep smiling")