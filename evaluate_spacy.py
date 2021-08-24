# import json
# import random
# import logging
# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_fscore_support
# from spacy.gold import GoldParse
# from spacy.scorer import Scorer
# from sklearn.metrics import accuracy_score
# import spacy

# tp=0
# tr=0
# tf=0
# ta=0
# c=0  


# def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
#     try:
#         training_data = []
#         lines=[]
#         with open(dataturks_JSON_FilePath, 'r',encoding="utf8") as f:
#             lines = f.readlines()

#         for line in lines:
#             data = json.loads(line)
#             text = data['content']
#             entities = []
#             for annotation in data['annotation']:
#                 #only a single point in text annotation.
#                 point = annotation['points'][0]
#                 labels = annotation['label']
#                 # handle both list of labels or a single label.
#                 if not isinstance(labels, list):
#                     labels = [labels]

#                 for label in labels:
#                     #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
#                     entities.append((point['start'], point['end'] + 1 ,label))


#             training_data.append((text, {"entities" : entities}))
#         # print(training_data)
#         return training_data
#     except Exception as e:
#         logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
#         return None

# examples=convert_dataturks_to_spacy("D:\\Downloads\\Rental agreements.json")
# # print(examples)
# nlp=spacy.load("D:\BE Project New")
# for text,annot in examples:

#     f=open("resume"+str(c)+".txt","w")
#     doc_to_test=nlp(text)
#     d={}
#     for ent in doc_to_test.ents:
#         d[ent.label_]=[]
#     for ent in doc_to_test.ents:
#         d[ent.label_].append(ent.text)
#     # print(d)
#     for i in set(d.keys()):

#         f.write("\n\n")
#         f.write(i +":"+"\n")
#         for j in set(d[i]):
#             f.write(j.replace('\n','')+"\n")
#     d={}
#     for ent in doc_to_test.ents:
#         d[ent.label_]=[0,0,0,0,0,0]
#     for ent in doc_to_test.ents:
#         doc_gold_text= nlp.make_doc(text)
#         gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
#         y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
#         y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
#         if(d[ent.label_][0]==0):
#             #f.write("For Entity "+ent.label_+"\n")   
#             #f.write(classification_report(y_true, y_pred)+"\n")
#             (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
#             a=accuracy_score(y_true,y_pred)
#             d[ent.label_][0]=1
#             d[ent.label_][1]+=p
#             d[ent.label_][2]+=r
#             d[ent.label_][3]+=f
#             d[ent.label_][4]+=a
#             d[ent.label_][5]+=1
#     c+=1
# print("-----------")
# print(d[ent.label_])
# print("-----------")
# for i in d:
#     print("\n For Entity "+i+"\n")
#     print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
#     print("Precision : "+str(d[i][1]/d[i][5]))
#     print("Recall : "+str(d[i][2]/d[i][5]))
#     print("F-score : "+str(d[i][3]/d[i][5]))








import json
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
import logging

def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r',encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))
        # print(training_data)
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

examples=convert_dataturks_to_spacy("D:\\Downloads\\Evaluate_spaCy.json")
# print(examples)
ner_model = spacy.load("D:\BE Project New") # for spaCy's pretrained use 'en_core_web_sm'
results = evaluate(ner_model, examples)
print(results)