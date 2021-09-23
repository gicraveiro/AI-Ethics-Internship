# MANIPULATING FACEBOOK SOURCED DATASET ON PRIVACY

import pdfx
import spacy
import nltk
import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS
#from nltk.corpus import stopwords
#nltk.download('stopwords')
file_input_path_general = 'Facebook/Privacy/TargetCompanySourced/' # global
# extracted from the lab, git repository: https://github.com/esrel/NLU.Lab.2021/blob/master/notebooks/corpus.ipynb
def nbest(d, n=5):
    """
    get n max values from a dict
    :param d: input dict (values are numbers, keys are strings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

# REMOVE STOP WORDS FUNCTION
def stop_words_removal(tokens,output_file_path):
    output_file = open(output_file_path, 'w')
    stop_word_file="extras/english.stop.txt"
    stop_word_f=open(stop_word_file,'r', encoding='utf-8')

    stop_words = (stop_word_f.read()).split()
    #doc = nlp(corpus)
    #tokens = [t for t in doc.text.split()]

    filtered_lexicon = (set(tokens).difference(stop_words))
    
    print("Stop word removal\n","Size of original lexicon:", len(tokens), "\nSize of filtered lexicon:",len(filtered_lexicon))#, file=output_file)

    for word in filtered_lexicon:
        print(word)#, file=output_file)

    stop_word_f.close()
    output_file.close()

    return tokens
    #return filtered_lexicon

def stop_words_removal_dict(dict,output_file_path):
    output_file = open(output_file_path, 'w')
    stop_word_file="extras/english.stop.txt"
    stop_word_f=open(stop_word_file,'r', encoding='utf-8')

    stop_words = (stop_word_f.read()).split()
    #doc = nlp(corpus)
    #tokens = [t for t in doc.text.split()]

    print(type(dict))
    print(dict)

    #filtered_lexicon = (set(dict).difference(stop_words))
    
    #print("Stop word removal\n","Size of original lexicon:", len(dict), "\nSize of filtered lexicon:",len(filtered_lexicon))#, file=output_file)

    #for word in filtered_lexicon:
    #    print(word)#, file=output_file)

    stop_word_f.close()
    output_file.close()

    return dict
    #return filtered_lexicon


def compute_stats(tokens,filename, path):
   
    output_file = open(path, 'w')
    print("\n"+filename+"\n", file=output_file)

    keywords_present = []

    freq = nltk.FreqDist(tokens)
    print('Total number of tokens:', len(tokens), file=output_file)
    # TO DO: TOTAL NUMBER OF UTTERANCES (SENTENCES)
    print('Size of Lexicon:', len(freq), file=output_file)
    
    #freq = stop_words_removal_dict(freq, 'output/'+file_input_path_general+'/'+filename+'/Stats.txt')

    plt.ion()
    graph = freq.plot(20, cumulative=False)#.invert_xaxis()
    plt.savefig('output/'+file_input_path_general+filename+'/Graph.png')  
    plt.clf() # cleans previous graph
    plt.ioff()

    freq = nbest(freq,len(freq))
    stop_word_file="extras/english.stop.txt"
    stop_word_f=open(stop_word_file,'r', encoding='utf-8') 
    stopwords = (stop_word_f.read()).split()

    print('\nTokens that appear in the file, alongside with their frequencies:', file=output_file)
    for key, val in freq.items():
        flag = 0
        for stopword in stopwords:
            if (key.lower() == stopword.lower()):
                flag = 1
        if(flag != 1): 
            print(str(key) + ':' + str(val), file=output_file)
        for keyword in keywords:
            if (keyword.lower() == key.lower()):
                keywords_present.append(str(key) + ':' + str(val))

    print("\nKeywords that appear in the file, alongside with their frequencies:", file=output_file)
    for keyword in keywords_present:
        print(keyword, file=output_file)
    
    output_file.close()

def process_document(title):
    input_file = pdfx.PDFx('data/'+file_input_path_general+title+'.pdf')
    input_file = input_file.get_text()
    doc = nlp(input_file)
    tokens = [t for t in doc.text.split()]
    #tokens = stop_words_removal(tokens,'output/'+file_input_path_general+'/'+title+'/Stats.txt') # filtering stop words
    compute_stats(tokens,title, 'output/'+file_input_path_general+'/'+title+'/Stats.txt') 

#####
#  MAIN 
#####

nlp = spacy.load('en_core_web_sm')

# CREATING GENERAL OUTPUT FILE
#output_file = open('output/PrivacyTargetCompanySourcedStats.txt', 'w')

# KEYWORDS 
output_file = open('output/Facebook/Privacy/Keywords.txt', 'w')
keywords_file = open('data/Keywords/Privacy KeyWords Formatted.docx.txt', "r", encoding='utf-8-sig')
keywords = keywords_file.read()
keywords = keywords.split(", ")
keywords_file.close()

print("Descriptive Statistics of Facebook Sourced Files on Privacy", file=output_file)
print("\nKeywords:\n",file=output_file)
for keyword in keywords:
    print(keyword, file=output_file)

process_document('CookiesPolicy') 
process_document('DataPolicy')
process_document('General Info ProtectingPrivacyAndSecurity')
process_document('OpenSourcePrivacyPolicy')
process_document('TermsOfService')

# TO DO: PUT COUNT OF THE SHOWN WORDS UP IN THE FILE

# MEETING: ASK WHAT KINDS OF INFO/DESCRIPTIVE STATISTICS WE WANT TO OBTAIN THAT WE DONT ALREADY HAVE
# COULD APPLY FREQUENCY CUT-OFF - IS IT WORTH IT? GIVEN THAT PRIVACY IS SAID ONCE MAYBE, I COULD SAY NO BUT TITLE MAYBE SHOULDNT COUNT, BUT ALSO HOW TO RULE OUT/MAKE SURE IT IS THE TITLE THAT IS BEING CUT OFF
# TO DO: PLOT ALL OF THE GRAPHS TOGETHER TOO BUT READABLY
# TO DO: TITLE TO THE GRAPHS
# TO DO: N GRAMS
# TO DO: FIND A WAY TO CHECK MEMORY LEAKS - ASK PROF RICCARDI?
# TO DO: MODIFY GRAPH SO THAT THE FULL WORDS CAN BE READ

# TO DO: FIGURE OUT HOW TO DEAL WITH THE COMMAS ',' AND PUNCTUATION THAT ARE BEING SEEING AS PART OF A TOKEN

#output_file.close()