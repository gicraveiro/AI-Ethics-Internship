# MANIPULATING FACEBOOK SOURCED DATASET ON PRIVACY
import os
import pdfx
import spacy
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')

file_input_path_general = 'Facebook/Privacy/' # global

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

""" def stop_words_removal(tokens,output_file_path):
    output_file = open(output_file_path, 'w')
    stop_word_file="extras/english.stop.txt"
    stop_word_f=open(stop_word_file,'r', encoding='utf-8')

    stop_words = (stop_word_f.read()).split()

    filtered_lexicon = (set(tokens).difference(stop_words))
    
    print("Stop word removal\n","Size of original lexicon:", len(tokens), "\nSize of filtered lexicon:",len(filtered_lexicon))#, file=output_file)

    for word in filtered_lexicon:
        print(word)#, file=output_file)

    stop_word_f.close()
    output_file.close()

    return tokens
    #return filtered_lexicon

 """
def compute_stats(tokens,filename): #, path):
   
    stats_path = 'output/'+file_input_path_general+filename+'/Stats.txt'
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    output_file = open(stats_path, 'w')
    print("\n"+filename+"\n", file=output_file)

    keywords_present = []
    #stop_word_file="data/Utils/english.stopwords.txt"
    #stop_word_f=open(stop_word_file,'r', encoding='utf-8') 
    #stopwords = (stop_word_f.read()).split()
    #stop_word_f.close()

    freq = nltk.FreqDist(tokens)
    print('Total number of tokens:', len(tokens), file=output_file)
    # TO DO: TOTAL NUMBER OF UTTERANCES (SENTENCES)
    print('Size of Lexicon:', len(freq), file=output_file)
    #filtered_lexicon = (set(tokens).difference(stopwords))
    #print("\nWith stop word removal","\nSize of original corpus:", len(tokens), "\nSize of filtered corpus:",len(filtered_lexicon), file=output_file)
    
    plt.ion()
    graph = freq.plot(20, cumulative=False) # TO DO: CHANGE GRAPH SO ALL WORDS BECOME READABLE
    plt.savefig('output/'+file_input_path_general+filename+'/Graph.png')  
    plt.clf() # cleans previous graph
    plt.ioff()
    #print(freq.tabulate(),'output/'+file_input_path_general+filename+'/Chart')

    freq = nbest(freq,len(freq))
    

    print('\nTokens that appear in the file, alongside with their frequencies:', file=output_file)
    for key, val in freq.items():
        flag = 0
        #for stopword in stopwords: # TO DO: OPTIMIZE, ACTUALLY FILTERING CORPUS INSTEAD OF FILTERING OUTPUT
        #    if (key.lower() == stopword.lower()):
        #        flag = 1
        if(flag != 1): 
            print(str(key) + ':' + str(val), file=output_file)
        for keyword in keywords:
            if (keyword.lower() == key.lower()):
                keywords_present.append(str(key) + ':' + str(val))

    print("\nKeywords that appear in the file, alongside with their frequencies:", file=output_file)
    for keyword in keywords_present:
        print(keyword, file=output_file)
    
    output_file.close()

def process_document(title, source):
    input_file = pdfx.PDFx('data/'+file_input_path_general+source+title+'.pdf') # TO DO: OPTIMIZE PATH, GET IT STRAIGHT FROM PARAMETER INSTEAD OF CALCULATING IT AGAIN
    input_file = input_file.get_text()

    doc = nlp(input_file)
    #tokens = [t for t in doc.text.split()]
    tokens = [token.text for token in doc if not token.is_space if not token.is_punct if not token.text.lower() in stopwords.words()]
    #print(tokens)
    #tokens = stop_words_removal(tokens,'output/'+file_input_path_general+'/'+title+'/Stats.txt') # filtering stop words
    compute_stats(tokens,source+title)#, 'output/'+file_input_path_general+source+title+'/Stats.txt') 

def analyse_folder(source):
    path='data/'+file_input_path_general+source
    for filename in os.listdir(path):
        file_name, file_extension = os.path.splitext(filename)
        process_document(file_name, source)

#####
#  MAIN 
#####

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("merge_noun_chunks")
nlp.add_pipe("merge_entities")

# KEYWORDS 
output_file = open('output/Facebook/Privacy/Keywords.txt', 'w')
keywords_file = open('data/Utils/PrivacyKeyWords2.txt', "r", encoding='utf-8-sig')
keywords = keywords_file.read()
keywords = keywords.split(", ")
keywords_file.close()

print("Descriptive Statistics of Facebook Sourced Files on Privacy", file=output_file)
print("\nKeywords:\n",file=output_file)
for keyword in keywords:
    print(keyword, file=output_file)

#### PERFORM DESCRIPTIVE STATISTICS ON ALL DATA
for foldername in os.listdir('data/'+file_input_path_general):
    print(foldername)
    analyse_folder(foldername+'/')


#################
## COMMENTS ON PROJECT PROGRESS

# SOLVE PROBLEM - KEYWORDS ARE NOT APPEARING

# A - CORPUS PRE-PROCESSING

# LEMMATIZATION, STEMMING - I THINK IT'S A GOOD IDEA BUT WE SHOULD CHECK THE KEYWORDS
# EXPANDING ABBREVIATIONS?
# TO DO: FIGURE OUT HOW TO DEAL WITH THE COMMAS ',' AND PUNCTUATION THAT ARE BEING SEEING AS PART OF A TOKEN

# B - TEST WITH DIFFERENT FILES -> DONE, NOT MUCH HELP...

# ACADEMIC PAPERS check
# NEWS check
# EVALUATIONS FROM OTHER SOURCES check

# C - IMPROVE KEYWORDS LIST

# CONSIDER MORE THAN ONE-WORD TOKENS!!! check
# PREPROCESSING LIKE LEMMATIZATION AND STEMMING
# INCLUDING MORE RELEVANT TERMS BY:
        # FREQUENCY ANALYSIS AT DIFFERENT DOCUMENTS
        # MANUALLY READING SOURCES AND REPORTING


# EXTRAS TO-DOS TO MAKE STATS PRETTIER/MORE COMPLETE

# TO DO: PLOT ALL OF THE GRAPHS TOGETHER TOO BUT READABLY and with no stop words!!!!
# TO DO: TITLE TO THE GRAPHS
# TO DO: N GRAMS
# TO DO: MODIFY GRAPH SO THAT THE FULL WORDS CAN BE READ

# MEETING: ASK WHAT KINDS OF INFO/DESCRIPTIVE STATISTICS WE WANT TO OBTAIN THAT WE DONT ALREADY HAVE
# COULD APPLY FREQUENCY CUT-OFF - IS IT WORTH IT? GIVEN THAT PRIVACY IS SAID ONCE MAYBE, I COULD SAY NO BUT TITLE MAYBE SHOULDNT COUNT, BUT ALSO HOW TO RULE OUT/MAKE SURE IT IS THE TITLE THAT IS BEING CUT OFF

# LOWERCASING? probably not needed
# NUMBERS TO WORDS? REMOVE NUMBERS? probably not needed
# EXPANDING ABBREVIATIONS? 
# READING OUT DATES? probably not needed
# TO DO: FIGURE OUT HOW TO DEAL WITH THE COMMAS ',' AND PUNCTUATION THAT ARE BEING SEEING AS PART OF A TOKEN!!!!
# TO DO: FIND A WAY TO CHECK MEMORY LEAKS - ASK PROF RICCARDI?
# AFTER ORGANIZING THIS ALL -> MOVE ON TO DEPENDENCY PARSING
