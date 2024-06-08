import streamlit as st
import joblib
#Import the libraries related to Building a Ml model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import warnings
#Import the libraries which is related to text
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import tqdm
from tqdm import tqdm , tqdm_notebook
tqdm.pandas()
warnings.filterwarnings("ignore") 
stemmer = PorterStemmer() #initialize with inbuilt stemmatization
lemmatizer = WordNetLemmatizer() #initialize with inbuilt lemmatization


def preproces(raw_text, flag):

    sentence = re.sub("[^a-zA-Z]"," ",str(raw_text)) #Extraction the texy

    sentence = sentence.lower() #Sentence into lower letters
    
    tokens = sentence.split() #split the sentence into tokens
    
    clean_tokens = [token for token in tokens if token not in stopwords.words("english")] #remove the stopwords and keep the text is important

    if flag.lower=="stem":
        clean_tokens = [stemmer.stem(word) for word in clean_tokens] #apply stemmatization

    else:
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens] #apply lemmatization

    return pd.Series([" ".join(clean_tokens),len(clean_tokens)] )
    
st.title("Sentiment Analysis")
model = joblib.load("C:\\Users\\Hello\\Prudent technologies\\Models\\Logistic Regression.pkl")
text_review = st.text_input(label="Text area")
Data = pd.DataFrame({"Review":text_review},index=[0])
if st.button("Review"):
    temp_df = Data['Review'].progress_apply(lambda text : preproces(text, flag="lemma"))
    columns=["Clean_Review","Len"]
    temp_df.columns=columns
    pred = model.predict(temp_df['Clean_Review'])
    st.header(pred[0])