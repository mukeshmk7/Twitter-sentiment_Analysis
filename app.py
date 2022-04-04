import os
import nltk
import numpy as np
import pandas as pd
import string
import re
import warnings
from joblib import dump, load
from flask import Flask, render_template, request, redirect
warnings.filterwarnings('ignore')

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = load('model.pkl')
vectorizer = load('vectorizer.pkl')
stopwords = stopwords.words('english')
punct = string.punctuation
lemma = WordNetLemmatizer()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    tweet = str(request.form.get('tweets'))
    final_tweet = tweet.lower()
    final_tweet = " ".join([lemma.lemmatize(word, pos='v') for word in final_tweet.split()])
    final_tweet = " ".join([word for word in final_tweet.split() if word not in stopwords])
    final_tweet = " ".join([word for word in final_tweet.split() if word not in punct])
    final_tweet = " ".join([re.sub(r'[0-9\.]+', '', word) for word in final_tweet.split()])
    final_tweet_vec = vectorizer.transform([final_tweet])
    output = model.predict(final_tweet_vec)
    print(output)
    if output == 'Positive':
        return render_template('predict.html', output="Positive Tweet")
    elif output == 'Negative':
        return render_template('predict.html', output="Negative Tweet")
    elif output == 'Neutral':
        return render_template('predict.html', output="Neutral Tweet")
    else:
        return render_template('predict.html', output="Irrelevant Tweet")

if __name__ == '__main__':
    app.run(debug=True)

