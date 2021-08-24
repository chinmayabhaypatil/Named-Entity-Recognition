import pandas as pd
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

df=pd.read_csv("D:\BE Project New\Classifier Data - Random.csv",usecols=["Sentence","Label"],encoding='utf-8')
# test=df['Sentence']
# # test=test.astype(np.int64)
# for text in test:
#     print(text.strip().lower())

punctuations = string.punctuation
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    return mytokens
i=0
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))
X=df['Sentence']
y=df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = BernoulliNB()

pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

pipe.fit(X_train,y_train)

predicted = pipe.predict(X_test)

print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

df=pd.read_csv("D:\BE Project New\Classifier Data - Contract.csv",usecols=["Sentence","Label"])
for user_ip in df['Sentence']:
    print(user_ip,pipe.predict(user_ip))
    print("\n\n\n")