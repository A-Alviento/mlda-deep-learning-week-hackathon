import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Page config
st.set_page_config(page_title="MLDA EEE 2022")

# Load dataset
data = pd.read_csv("./data/spam.csv", nrows = 100, encoding = "ISO-8859-1") 
data = data[["v1", "v2"]]

# Page layout
st.title("Spam Call Detector")
st.markdown("Here is a preview of the [dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)\
                that we used to train our spam detection model. There is about\
                5000 messages in English with the spam being marked as *spam*\
                and real messages being labeld as *ham*.")
st.dataframe(data)

st.markdown("The generated word clouds between the spam and real messages showcases\
                the difference between real and fake messages. Our machine learning\
                model will capitalize on the messages to predict whether a given\
                message is *spam* or *ham*.")
st.image("./data/ham.png", caption = "Spam messages")
st.image("./data/spam.png", caption = "Real messages")

# Load model at the end to make page loading time faster
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model

model = load_model("./data/model.h5")
tokenizer = pickle.load(open("./data/tokenizer.pickle", "rb"))
maxlen = 100

spam = st.text_input("Insert some spam to test it out!")
if st.button("Predict"):
    
    with st.spinner("Processing..."):
        output = ""
        for c in spam:
          if (c.isalpha() or c == " "):
            output += c
          else:
            output += " "
        x = [output]
        
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=100)
        prediction = str(model.predict(x)[0][0])
        
       
        if prediction >= 0.5:
            st.header("This is spam!")
        else:
           st.header("This is ham!")
  
