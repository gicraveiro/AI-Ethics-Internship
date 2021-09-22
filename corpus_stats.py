# MANIPULATING FACEBOOK SOURCED DATASET ON PRIVACY

import pdfx
import spacy
import nltk

# extracted from the lab, git repository: https://github.com/esrel/NLU.Lab.2021/blob/master/notebooks/corpus.ipynb
def nbest(d, n=5):
    """
    get n max values from a dict
    :param d: input dict (values are numbers, keys are stings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def compute_stats(file):

    doc = nlp(file)
    tokens = [t for t in doc.text.split()]

    freq = nltk.FreqDist(tokens)
    print('Total number of tokens:', len(tokens))
    # TO DO: TOTAL NUMBER OF UTTERANCES (SENTENCES)
    print('Size of Lexicon:', len(freq))

    #print(freq.keys())
    for key, val in freq.items():
        #print(str(key) + ':' + str(val))
        for keyword in keywords:
            if (keyword.lower() == key.lower()):
                print(str(key) + ':' + str(val))

    freq.plot(20, cumulative=False)
    print(nbest(freq,len(freq)))

nlp = spacy.load('en_core_web_sm')

# KEYWORDS 

keywords = open('data/Keywords/Privacy KeyWords Formatted.docx.txt', "r", encoding='utf-8-sig')
keywords = keywords.read()
keywords = keywords.split(", ")
print(keywords)

# TRANSFORMING ALL PDFS INTO TEXT

cookies_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/CookiesPolicy.pdf')
cookies_policy = cookies_policy.get_text()
#print(cookies_policy)
print("\nCookie Policy Stats\n")
compute_stats(cookies_policy)

data_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/DataPolicy.pdf')
data_policy = data_policy.get_text()
#print(data_policy)
print("\nData Policy Stats\n")
compute_stats(data_policy)

gen_info_privacy = pdfx.PDFx('data/Facebook/TargetCompanySourced/General Info ProtectingPrivacyAndSecurity.pdf')
gen_info_privacy = gen_info_privacy.get_text()
#print(gen_info_privacy)
print("\nGeneral Info on Privacy Stats\n")
compute_stats(gen_info_privacy)

open_source_privacy_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/OpenSourcePrivacyPolicy.pdf')
open_source_privacy_policy = open_source_privacy_policy.get_text()
#print(open_source_privacy_policy)
print("\nOpen Source Privacy Policy Stats\n")
compute_stats(open_source_privacy_policy)

terms_of_service = pdfx.PDFx('data/Facebook/TargetCompanySourced/TermsOfService.pdf')
terms_of_service = terms_of_service.get_text()
#print(terms_of_service)
print("\nTerms of Service Stats\n")
compute_stats(terms_of_service)