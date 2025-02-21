# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pandas as pd
import streamlit as st
import re
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
import numpy as np
lemmatizer=WordNetLemmatizer()

from PIL import Image

classifier_pkl = open("classifier.pkl","rb")
classifier=pickle.load(classifier_pkl)

model_pkl = open("model.pkl","rb")
model=pickle.load(model_pkl)


def welcome():
    return "Welcome All"

def avg_word2vec(doc):
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis = 0)

def predict_spam_or_ham(input_text):
    
    """Let's predict if the message is a Spam message ot not
    """
    sms = re.sub(r'[^a-zA-Z]', ' ', input_text)
    sms = sms.lower()
    sms = sms.split()
    sms = [lemmatizer.lemmatize(word) for word in sms]
    sms = ' '.join(sms)
    sms_message = simple_preprocess(sms)
    out = avg_word2vec(sms_message)
    print(type(out))
    test_df = pd.DataFrame()
    test_df = pd.concat([test_df.astype('float64'), pd.DataFrame(out.reshape(1, -1))], ignore_index=True)
    answer = classifier.predict(test_df)
    if answer[0] == 0:
        return "A Spam!!"

    else:
        return "Not A Spam"

def main():
    st.title("Spam Detection")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Spam detection ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    input_text = st.text_input("Variance","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_spam_or_ham(input_text)
    st.success('The message body which you enter is {}'.format(result))

if __name__=='__main__':
    main()
    
    
    