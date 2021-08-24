import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier,LogisticRegression
# from sklearn.externals import joblib
import joblib
from sklearn.feature_extraction import text
import string

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB


dig = set(string.digits)
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space

df = pd.read_csv("D:\BE Project New\Classifier Data - Random.csv")
sentences = df['Sentence']
y = df['Label']

# new_sentences = []
# for sentence in sentences:
#     sentence = sentence.lower()
#     sentence = sentence.translate(translator)
#     sentence = ''.join([i for i in sentence if i not in dig])
#     new_sentences.append(sentence)

vectorizer = CountVectorizer()
vectorizer.fit(sentences)
X_train = vectorizer.transform(sentences)

# save the model to disk
vect_filename = 'final_isa_vectorizer.sav'
joblib.dump(vectorizer, vect_filename)

#classifier = SGDClassifier(loss='log')
classifier=NuSVC(probability=True)
classifier.fit(X_train, y)

# # save the model to disk
model_filename = 'final_isa_classifier.sav'
joblib.dump(classifier, model_filename)
print("ok")
