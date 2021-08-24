import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction import text
import string
import docxpy
from nltk.tokenize import sent_tokenize
import testSpacy

def intent_classifier(user_ip_statement):

    # digits = set(string.digits)

    # Loading the Count Vectorizer
    vect_filename = 'final_isa_vectorizer.sav'
    loaded_vectorizer = joblib.load(vect_filename)

    # Loading the Logistic Regression model
    model_filename = 'final_isa_classifier.sav'
    loaded_model = joblib.load(model_filename)

    # User Input Text
    input_text = user_ip_statement


    vectorized_input = loaded_vectorizer.transform([input_text])
    result_prob = loaded_model.predict_proba(vectorized_input)
    result=0
    
    if (result_prob[0][1] >=0.3):
        result=1
    
    if(str(result==1)):
                return input_text
    return ''

def inputClassifier(all_text_array):
        filteredSentences=''
        for i in range(len(all_text_array)):
                try:
            # user_ip = input("Enter Text:")
                        filteredSentences=filteredSentences+' '+intent_classifier(all_text_array[i])

                except Exception as e:
                        print("Error: " + str(e))
        print(filteredSentences)
        testSpacy.predict(filteredSentences)
