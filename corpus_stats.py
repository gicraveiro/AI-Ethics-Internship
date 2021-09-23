# MANIPULATING FACEBOOK SOURCED DATASET ON PRIVACY

import pdfx
import spacy
import nltk
import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS
#from nltk.corpus import stopwords
#nltk.download('stopwords')

# extracted from the lab, git repository: https://github.com/esrel/NLU.Lab.2021/blob/master/notebooks/corpus.ipynb
def nbest(d, n=5):
    """
    get n max values from a dict
    :param d: input dict (values are numbers, keys are strings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def compute_stats(file,filename, path):
   
    output_file = open(path, 'w')
    print("\n"+filename+"\n", file=output_file)

    doc = nlp(file)
    tokens = [t for t in doc.text.split()]
    keywords_present = []

    freq = nltk.FreqDist(tokens)
    print('Total number of tokens:', len(tokens), file=output_file)
    # TO DO: TOTAL NUMBER OF UTTERANCES (SENTENCES)
    print('Size of Lexicon:', len(freq), file=output_file)
    
    plt.ion()
    graph = freq.plot(20, cumulative=False) 
    plt.savefig('output/Privacy/FacebookSourced/'+filename+'/Graph.png') # TO DO: GENERALIZE ETHIC THEME + SOURCE 
    plt.clf() # cleans previous graph
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
    
    output_file.close()
    
#####
#  MAIN 
#####

nlp = spacy.load('en_core_web_sm')

# CREATING GENERAL OUTPUT FILE
#output_file = open('output/PrivacyFacebookSourcedStats.txt', 'w')

# KEYWORDS 
output_file = open('output/Privacy/Keywords.txt', 'w')
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
compute_stats(cookies_policy, 'CookiesPolicy', 'output/Privacy/FacebookSourced/CookiesPolicy/Stats.txt')

data_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/DataPolicy.pdf')
data_policy = data_policy.get_text()
#print(data_policy)
compute_stats(data_policy,'DataPolicy', 'output/Privacy/FacebookSourced/DataPolicy/Stats.txt')

gen_info_privacy = pdfx.PDFx('data/Facebook/TargetCompanySourced/General Info ProtectingPrivacyAndSecurity.pdf')
gen_info_privacy = gen_info_privacy.get_text()
#print(gen_info_privacy)
compute_stats(gen_info_privacy, 'General Info ProtectingPrivacyAndSecurity', 'output/Privacy/FacebookSourced/General Info ProtectingPrivacyAndSecurity/Stats.txt')

open_source_privacy_policy = pdfx.PDFx('data/Facebook/TargetCompanySourced/OpenSourcePrivacyPolicy.pdf')
open_source_privacy_policy = open_source_privacy_policy.get_text()
#print(open_source_privacy_policy)
compute_stats(open_source_privacy_policy, 'OpenSourcePrivacyPolicy', 'output/Privacy/FacebookSourced/OpenSourcePrivacyPolicy/Stats.txt')

terms_of_service = pdfx.PDFx('data/Facebook/TargetCompanySourced/TermsOfService.pdf')
terms_of_service = terms_of_service.get_text()
#print(terms_of_service)
compute_stats(terms_of_service, 'TermsOfService', 'output/Privacy/FacebookSourced/TermsOfService/Stats.txt')

# TO DO: REMOVE STOP WORDS FUNCTION

stop_word_file="extras/english.stop.txt"
stop_word_f=open(stop_word_file,'r', encoding='utf-8')

stop_words = (stop_word_f.read()).split()
print(stop_words)
#print(cookies_policy)
doc = nlp(cookies_policy)
tokens = [t for t in doc.text.split()]
#print(type(cookies_policy))
#print(len(tokens))

filtered_lexicon = (set(tokens).difference(stop_words))
print(filtered_lexicon, len(filtered_lexicon), len(tokens))

stop_word_f.close()
# TO DO: PUT COUNT OF THE SHOWN WORDS UP IN THE FILE

# MEETING: ASK WHAT KINDS OF INFO/DESCRIPTIVE STATISTICS WE WANT TO OBTAIN THAT WE DONT ALREADY HAVE
# COULD APPLY FREQUENCY CUT-OFF - IS IT WORTH IT? GIVEN THAT PRIVACY IS SAID ONCE MAYBE, I COULD SAY NO BUT TITLE MAYBE SHOULDNT COUNT, BUT ALSO HOW TO RULE OUT/MAKE SURE IT IS THE TITLE THAT IS BEING CUT OFF
# TO DO: PLOT ALL OF THE GRAPHS TOGETHER TOO BUT READABLY
# TO DO: TITLE TO THE GRAPHS
# TO DO: ADD THE TRANSFORMATION OF FILES TO A FUNCTION
# TO DO: N GRAMS
# TO DO: FIND A WAY TO CHECK MEMORY LEAKS - ASK PROF RICCARDI?


#output_file.close()