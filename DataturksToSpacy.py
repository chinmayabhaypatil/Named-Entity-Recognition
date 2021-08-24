import logging
import json
import spacy
import random

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

# data=convert_dataturks_to_spacy("D:\\BE Project New\\News Articles.json")
# print(type(data))








def train_spacy():
    TRAIN_DATA = convert_dataturks_to_spacy("D:\\BE Project New\\News Articles.json")
    # with open("D:\\BE Project New\\spacyConverted.json",'r', encoding='utf-8') as f:
        # TRAIN_DATA=f.read()
    nlp = spacy.blank('en') 


    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)


    # add labels
    for _,annotations in TRAIN_DATA:
        annotations=(annotations)
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(30):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    nlp.to_disk("D:\BE Project New")
    nlp2=spacy.load("D:\BE Project New")
    # do prediction
    doc = nlp("RESIDENTIAL RENTAL AGREEMENT This agreement made at Mumbai, Maharashtra on this 23rd September 2019 between Shriram Iyer, residing at 19, Riddhi Indl Estate, Gaurai Pada, Gokhivare, Mumbai, Maharashtra, 401205 hereinafter referred to as the `LESSOR` of the One Part AND Poshita  Persad, residing at 1/202, Huseni Ladaka Bazaar, Bellasis Road, Mumbai, Maharashtra, 400008 hereinafter referred to as the `LESSEE` of the other Part; WHEREAS the Lessor is the lawful owner of, and otherwise well sufficiently entitled to A-142 Ghatkopar Industrial Esta, Lb Shastri Marg, Ghatkopar (West), Mumbai, Maharashtra, 400086 falling in the category, Residential Property and comprising of 2 Bedrooms, 2 Bathrooms, 1 Carparks with an extent of 950 Square Feet hereinafter referred to as the `said premises`; AND WHEREAS at the request of the Lessee, the Lessor has agreed to let the said premises to the tenant for a term of 24 months commencing from 30th September 2019 in the manner hereinafter appearing. ")
    print ("Entities= " + str(["" + str(ent.text) + "_" + str(ent.label_) for ent in doc.ents]))
    doc2 = nlp2("RESIDENTIAL RENTAL AGREEMENT This agreement made at Mumbai, Maharashtra on this 23rd September 2019 between Shriram Iyer, residing at 19, Riddhi Indl Estate, Gaurai Pada, Gokhivare, Mumbai, Maharashtra, 401205 hereinafter referred to as the `LESSOR` of the One Part AND Poshita  Persad, residing at 1/202, Huseni Ladaka Bazaar, Bellasis Road, Mumbai, Maharashtra, 400008 hereinafter referred to as the `LESSEE` of the other Part; WHEREAS the Lessor is the lawful owner of, and otherwise well sufficiently entitled to A-142 Ghatkopar Industrial Esta, Lb Shastri Marg, Ghatkopar (West), Mumbai, Maharashtra, 400086 falling in the category, Residential Property and comprising of 2 Bedrooms, 2 Bathrooms, 1 Carparks with an extent of 950 Square Feet hereinafter referred to as the `said premises`; AND WHEREAS at the request of the Lessee, the Lessor has agreed to let the said premises to the tenant for a term of 24 months commencing from 30th September 2019 in the manner hereinafter appearing. ")
    print ("Entities= " + str(["" + str(ent.text) + "_" + str(ent.label_) for ent in doc2.ents]))

train_spacy()
