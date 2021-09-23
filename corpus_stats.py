# MANIPULATING FACEBOOK SOURCED DATASET ON PRIVACY

import pdfx
import spacy
import nltk
import matplotlib.pyplot as plt

# extracted from the lab, git repository: https://github.com/esrel/NLU.Lab.2021/blob/master/notebooks/corpus.ipynb
def nbest(d, n=5):
    """
    get n max values from a dict
    :param d: input dict (values are numbers, keys are strings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def compute_stats(file,filename):

    doc = nlp(file)
    tokens = [t for t in doc.text.split()]
    keywords_present = []

    freq = nltk.FreqDist(tokens)
    print('Total number of tokens:', len(tokens), file=output_file)
    # TO DO: TOTAL NUMBER OF UTTERANCES (SENTENCES)
    print('Size of Lexicon:', len(freq), file=output_file)
    
    plt.ion()
    graph = freq.plot(20, cumulative=False) # TO DO: ADD GRAPHS TO FILE
    plt.savefig('output/'+filename+'Graph.png')
    plt.ioff()

    freq = nbest(freq,len(freq)) 

    print('\nTokens that appear in the file, alongside with their frequencies:', file=output_file)
    for key, val in freq.items():
        print(str(key) + ':' + str(val), file=output_file)
        for keyword in keywords:
            if (keyword.lower() == key.lower()):
                keywords_present.append(str(key) + ':' + str(val))

    print("\nKeywords that appear in the file, alongside with their frequencies:", file=output_file)
    for keyword in keywords_present:
        print(keyword, file=output_file)
    
#####
#  MAIN 
#####

nlp = spacy.load('en_core_web_sm')

# CREATING OUTPUT FILE

output_file = open('output/PrivacyFacebookSourcedStats.txt', 'w')

# KEYWORDS 

keywords_file = open('data/Keywords/Privacy KeyWords Formatted.docx.txt', "r", encoding='utf-8-sig')
keywords = keywords_file.read()
keywords = keywords.split(", ")
keywords_file.close()
print("Descriptive Statistics of Facebook Sourced Files on Privacy", file=output_file)
print("\nKeywords:\n",file=output_file)
for keyword in keywords:
    print(keyword, file=output_file)


# TRANSFORMING ALL PDFS INTO TEXT AND COMPUTING STATS

cookies_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/CookiesPolicy.pdf')
cookies_policy = cookies_policy.get_text()
#print(cookies_policy)
print("\nCookie Policy Stats\n", file=output_file)
compute_stats(cookies_policy, 'CookiesPolicy')

data_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/DataPolicy.pdf')
data_policy = data_policy.get_text()
#print(data_policy)
print("\nData Policy Stats\n", file=output_file)
compute_stats(data_policy,'DataPolicy')

gen_info_privacy = pdfx.PDFx('data/Facebook/TargetCompanySourced/General Info ProtectingPrivacyAndSecurity.pdf')
gen_info_privacy = gen_info_privacy.get_text()
#print(gen_info_privacy)
print("\nGeneral Info on Privacy Stats\n", file=output_file)
compute_stats(gen_info_privacy, 'General Info ProtectingPrivacyAndSecurity')

open_source_privacy_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/OpenSourcePrivacyPolicy.pdf')
open_source_privacy_policy = open_source_privacy_policy.get_text()
#print(open_source_privacy_policy)
print("\nOpen Source Privacy Policy Stats\n", file=output_file)
compute_stats(open_source_privacy_policy, 'OpenSourcePrivacyPolicy')

terms_of_service = pdfx.PDFx('data/Facebook/TargetCompanySourced/TermsOfService.pdf')
terms_of_service = terms_of_service.get_text()
#print(terms_of_service)
print("\nTerms of Service Stats\n", file=output_file)
compute_stats(terms_of_service, 'TermsOfService')

# TO DO: REMOVE STOP WORDS FUNCTION
# TO DO: SEPARATE OUTPUT OF EACH FILE INTO ITS OWN OUTPUT FILE 

output_file.close()