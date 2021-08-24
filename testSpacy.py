import spacy
import docxpy
from spacy import displacy
from spacy.scorer import Scorer
from spacy.gold import GoldParse
import logging
import json
import spacy
import random

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

# data=convert_dataturks_to_spacy("D:\\Downloads\\Rental agreements.json")
# # print(data)

def predict(doc):
##    doc=IsAClassifier_test.inputClassifier("D:\Documents\BE Project\Dataset\Agreement\Rent 2.docx")
    nlp=spacy.load("D:\BE Project New")
    ####
    doc = nlp(doc)

    ##doc=docxpy.process("D:\Documents\BE Project\Dataset\Agreement\Rent 3.docx")
    ##doc_ent=nlp(doc)
    ##s=''
    ##for para in doc.split("\n"):
    ##    s=s+" "+para
    ##doc_ent=nlp(s)
    ##print(s.replace("\n"," "))
    ##doc=nlp("This agreement made at Pune, Maharashtra on this 17th August 2018 between Valmiki Dhar residing at 10/1-a, Tilak Bhavan, Elphin Stone Road, Bopodi, Pune Maharashtra, 411003, hereinafter referred to as the `LESSOR` of the One Part AND Amar Sasthri residing at Bunglow No. 14, The Swing, Marve Road, Opp Nutan High School, Malad(West), Mumbai, Maharashtra, 400064 hereinafter referred to as the `LESSEE` of the other Part; WHEREAS the Lessor is the lawful owner of, and otherwise well sufficiently entitled to C/305, Gayatri Complex, Bibwewadi, Society, Pune, Maharashtra, 411037 falling in the category, Residential Property and comprising of 3 Bedrooms, 2 Bathrooms with an extent of 1350 Square feet hereinafter referred to as the `said premises`;AND WHEREAS at the request of the Lessee, the Lessor has agreed to let the said premises to the tenant for a term of 12 months commencing from 1st November 2018 in the manner hereinafter appearing. NOW THIS AGREEMENT WITNESSETH AND IT IS HEREBY AGREED BY AND BETWEEN THE PARTIES AS UNDER: 2.	That the lease hereby granted shall, unless cancelled earlier under any provision of this Agreement, remain in force for a period of 12 months. 11.	That in consideration of use of the said premises the Lessee agrees that he shall pay to the Lessor during the period of this agreement, a monthly rent at the rate of Rs. 11250 or Rupees Eleven Thousand Two Hundred and Fifty Only. The amount will be paid in advance on or before the date of 5th day of every English calendar month. 15.	That the Lessor shall be responsible for the payment of all taxes and levies pertaining to the said premises including but not limited to House Tax, Property Tax, other cesses, if any, and any other statutory taxes, levied by the Government or Governmental Departments. During the term of this Agreement, the Lessor shall comply with all rules, regulations and requirements of any statutory authority, local, state and central government and governmental departments in relation to the said premises.")

    colors = {"NAME": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
              "ADDRESS": "linear-gradient(90deg, #b3ffb3, #ffff00)",
              "DATE": "linear-gradient(90deg, #f97e7e, #ff0000)",
              "PERIOD (MONTHS)": "linear-gradient(90deg, #8edfeb, #40c8dd)",
              "AMOUNT": "linear-gradient(90deg, #e56c74, #f4bec2)",
              "PAYMENT DAY": "linear-gradient(90deg, #d92926, #26d9c7)"}
    options = {"ents": ["NAME","ADDRESS","DATE","PERIOD (MONTHS)","AMOUNT","PAYMENT DAY"], "colors": colors}



    displacy.serve(doc, style='ent',options=options)

# def evaluate(nlp, docs_golds, verbose=False):

#     scorer = Scorer()
#     docs, golds = zip(*docs_golds)
#     docs = list(docs)
#     golds = list(golds)
#     pipeline=[]
#     for name, pipe in pipeline:
#         if not hasattr(pipe, 'pipe'):
#             docs = (pipe(doc) for doc in docs)
#         else:
#             docs = pipe.pipe(docs, batch_size=256)
#     for doc, gold in zip(docs, golds):
#         if verbose:
#             print(doc)
#         scorer.score(doc, gold, verbose=verbose)
#     return scorer


    # scorer = Scorer()
    # for input_, annot in examples:
    #     text_entities = []
    #     for entity in annot.get('entities'):
    #         if ent in entity:
    #             text_entities.append(entity)
    #     doc_gold_text = nlp.make_doc(input_)
    #     gold = GoldParse(doc_gold_text, entities=text_entities)
    #     pred_value = nlp(input_)
    #     pred_value.ents = [e for e in pred_value.ents if e.label_ == ent]
    #     scorer.score(pred_value, gold)
    # return scorer.scores


# examples = [
#     ("Trump says he's answered Mueller's Russia inquiry questions \u2013 live",{"entities":[[0,5,"PERSON"],[25,32,"PERSON"],[35,41,"GPE"]]}),
#     ("Alexander Zverev reaches ATP Finals semis then reminds Lendl who is boss",{"entities":[[0,16,"PERSON"],[55,60,"PERSON"]]}),
#     ("Britain's worst landlord to take nine years to pay off string of fines",{"entities":[[0,7,"GPE"]]}),
#     ("Tom Watson: people's vote more likely given weakness of May's position",{"entities":[[0,10,"PERSON"],[56,59,"PERSON"]]}),
# ]

# nlp = spacy.load('D:\BE Project New')
# results = evaluate(nlp, data)
# print(results)

nlp=spacy.load("D:\BE Project New")
doc=nlp("with an extent of 800 Square Feet hereinafter referred to as the `said premises`;")
colors = {"NAME": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
              "ADDRESS": "linear-gradient(90deg, #b3ffb3, #ffff00)",
              "DATE": "linear-gradient(90deg, #f97e7e, #ff0000)",
              "PERIOD (MONTHS)": "linear-gradient(90deg, #8edfeb, #40c8dd)",
              "AMOUNT": "linear-gradient(90deg, #e56c74, #f4bec2)",
              "PAYMENT DAY": "linear-gradient(90deg, #d92926, #26d9c7)",
              "DAY OF WEEK": "linear-gradient(90deg, #d92926, #26d9c7)"}
options = {"ents": ["NAME","ADDRESS","DATE","PERIOD (MONTHS)","AMOUNT","PAYMENT DAY","DAY OF WEEK"], "colors": colors}
displacy.serve(doc, style='ent',options=options)