import spacy
import pdfx
from nltk.corpus import stopwords
import re

nlp = spacy.load('en_core_web_sm')
#input_file = "lala . 1 third-party organization isaca's cobit 5 3 yay"
#input_file = " including , but not limited to , operational procedures , management processes , structure of products , selection of business partners , employee training , and use of technology . 3 the entity shall describe its use of third-party cybersecurity risk management standards . 3 . 1\tthird-party cybersecurity risk management standards are defined as standards , frameworks , and/or guidance developed by a third-party with the explicit purpose of aiding companies in identifying cybersecurity threats , and/or preventing , responding to , and/or remediating cybersecurity incidents . 3 . 2 examples of third-party cybersecurity risk management standards include , but are not limited to: 3 . 2 . 1 the american institute of certified public accountants’ ( aicpa ) service organization controls ( soc ) for cybersecurity 3 . 2 . 2 isaca’s cobit 5\n3 . 2 . 3 iso/iec 27000-series 3 . 2 . 4 national institute of standards and technology’s ( nist ) framework for improving critical infrastructure cybersecurity , version 1 . 1 3 . 3 disclosure shall include , "
path="data/Guidelines/SASB-Privacy-Software_IT_Services_Standard_2018.pdf"
input_file = pdfx.PDFx(path) # TO DO: OPTIMIZE PATH, GET IT STRAIGHT FROM PARAMETER INSTEAD OF CALCULATING IT AGAIN
input_file = input_file.get_text()
doc = nlp(input_file)

input_file = re.sub("\s+", r"  ", input_file)
print("STEP -1","\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
input_file = re.sub("(\s+\-)", r" - ", input_file)
i = 0
print("STEP",i,"\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
i += 1
input_file = re.sub("([a-zA-Z]+)([0-9]+)", r"\1 \2", input_file)
print("STEP",i,"\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
i += 1
input_file = re.sub("([0-9]+)([a-zA-Z]+)", r"\1 \2", input_file)
print("STEP",i,"\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
i += 1
input_file = re.sub("([()!,;\.\?\[\]\|])", r" \1 ", input_file)
print("STEP",i,"\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
#i += 1
#input_file = re.sub("\t+"," ", input_file)
#print("STEP",i,"\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
#i += 1
input_file = re.sub("\s+"," ", input_file)
#input_file = re.sub(" "," ", input_file)
print("STEP",i,"\n",[token.text for token in nlp(input_file) if not token.is_space if not token.is_punct if not token.text in stopwords.words()],"\n")
i += 1
input_file = input_file.lower()

tokens = [token.text for token in doc if not token.is_space if not token.is_punct if not token.text in stopwords.words()]

print(tokens)