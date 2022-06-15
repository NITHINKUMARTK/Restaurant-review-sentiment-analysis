import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
st.title("RESTAURANT REVIEW SENTIMENT ANALYSIS")
st.write("Enter the review")
sample_review = st.text_input("")
pickle_in = open("restaurant.pkl","rb")
naive = pickle.load(pickle_in)

pickle_in = open("cv_transform.pkl","rb")
cv = pickle.load(pickle_in)


if st.button("Predict"):
	def predict_sentiment(sample_review):
		sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
		sample_review = sample_review.lower()
		sample_review_words = sample_review.split()
		sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
		ps = PorterStemmer()
		final_review = [ps.stem(word) for word in sample_review_words]
		final_review = ' '.join(final_review)
		temp = cv.transform([final_review]).toarray()
		return naive.predict(temp)

	if predict_sentiment(sample_review):
		st.success('This is a POSITIVE review.')
	else:
		st.success('OOPS!, This is a NEGATIVE review!')
  

  

  
  

 
  
 
 

  
  


