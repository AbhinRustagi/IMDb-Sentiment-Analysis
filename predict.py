import numpy as np
import keras
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import string

model = keras.models.load_model('model.h5')
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("feature.pkl", "rb")))

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getStemmedReview(reviews):
  review = reviews.lower()
  review = review.replace('<br /><br />', "")

  tokens = tokenizer.tokenize(review)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(token) for token in new_tokens]

  clean_review = ' '.join(stemmed_tokens)

  return clean_review

print("Hello!\n")

ans = 'y'

while ans=='y':

    sample = input("Please input a string!\n")

    stemmed_sample = [getStemmedReview(sample)]
    x = loaded_vec.fit_transform(stemmed_sample).toarray()

    prediction = model.predict(x)[0][0]

    if prediction >= 0.5:
        print("Positive! :)")
    else:
        print("Negative! :(")

    print("Do you wish to continue? (Y/N)\n")
    ans = input().lower()

    if ans=='y':
        continue
    elif ans=='n':
        break
    else:
        print("Not a valid key. Program rerun.")
