# MANIPULATING FACEBOOK SOURCED DATASET ON PRIVACY
import os
import pdfx
import spacy
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
#nltk.download('stopwords')

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

def plot_graph(tokens,path,title):
    freq = nltk.FreqDist(tokens) 
    plt.ion()
    graph = freq.plot(40, cumulative=False, title=title)
    plt.savefig(path, bbox_inches='tight') 
    # IF WE NEED TO RECREATE THE JOINT GRAPH, COMMENT THIS COMMAND TO STOP REFRESHING THE GRAPH :
    plt.clf() # cleans previous graph
    plt.ioff()
    return freq
def compute_stats(tokens, filename, output_file, gen_path): 
    # GRAPH OF WORD FREQUENCY
    #freq = nltk.FreqDist(tokens) # NOW LOCATED INSIDE THE FUNCTION
    graph_path='output/'+gen_path+'/'+filename+'/Graph.png'
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    freq = plot_graph(tokens,graph_path,filename)
    
    # TO DO: TOTAL NUMBER OF UTTERANCES (SENTENCES)
    print('Size of Lexicon:', len(freq), file=output_file)

    # ORDERING LEXICON BY FREQUENCY COUNT
    freq = nbest(freq,len(freq))
    
    # COUNTING KEYWORDS AND PRINTING EVERYTHING IN THE OUTPUT FILE
    print('\nTokens that appear in the file, alongside with their frequencies:', file=output_file)
    keywords_present = []
    for key, val in freq.items():
        print(str(key) + ':' + str(val), file=output_file)
        for keyword in keywords:
            if (keyword.lower() == key.lower()):
                keywords_present.append(str(keyword) + ':' + str(val))

    print("\nKeywords that appear in the file, alongside with their frequencies:", file=output_file)
    for keyword in keywords_present:
        print(keyword, file=output_file)

def string_search(document, index,keyword):
    counter=0
    index = document.find(keyword, index+1) #condition to prevent index from reaching the end needed?
    if(index != -1): 
        counter +=1
        counter += string_search(document,index,keyword)
    return counter

# reconstructs hyphen, slash and apostrophes
def reconstruct_hyphenated_words(corpus):
    i = 0
    while i < len(corpus):
        if((corpus[i].text == "-" or corpus[i].text == "/") and corpus[i].whitespace_ == ""): # identify hyphen ("-" inside a word)
            print("reconstruction 1", corpus[i-1]," ->", corpus[i-1:i+2] )
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+2]) # merge the first part of the word, the hyphen and the second part of the word            
        elif(corpus[i].text == "’s" and corpus[i-1].whitespace_ == ""):
            print("reconstruction 2", corpus[i-1]," ->",corpus[i-1:i+1] )
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+1])
            
        else: 
            i += 1
    return corpus

# noun chunks that correspond to keywords
def reconstruct_noun_chunks(corpus,keywords):
    i = 0
    #max_length = len(corpus)
    while i < len(corpus):
        counter = i
        token = corpus[i].text
        for keyword in keywords:
            kw_lower = keyword.lower()
            index = kw_lower.find(token)
            aux = index
            while (aux != -1 and counter < len(corpus)-1 and token != kw_lower):
                counter += 1
                token += ' '+corpus[counter].text  # +' ' 
                aux = kw_lower.find(token)
                if(aux == -1):
                    counter -=1
                    token = corpus[i].text
            if(i != counter):
                if(token == kw_lower): #str(corpus[i:counter+1])
                    print("reconstruction", token) #str(corpus[i:counter+1]))
                    with corpus.retokenize() as retokenizer:
                        retokenizer.merge(corpus[i:counter+1])
                    break 
                else: 
                    counter = i               
        if(i == counter):
            i += 1
    return corpus

def process_document(title, source_path,source,keywords):
    
    # CREATING OUTPUT FILES
    stats_path = 'output/'+source_path+'/'+title+'/Stats.txt'
    keyword_guide_path = 'output/'+source_path+'/'+title+'/KeywordGuide.txt'
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    os.makedirs(os.path.dirname(keyword_guide_path), exist_ok=True)
    output_file = open(stats_path, 'w')
    print("\n"+title+"\n", file=output_file)
    
    # READING AND MANIPULATING INPUT FILE
    #path = 'data/'+file_input_path_general+source+title+'.pdf'
    path = 'data/'+source_path+'/'+title+'.pdf' #'data/'+file_input_path_general+title+'.pdf'
    input_file = pdfx.PDFx(path) # TO DO: OPTIMIZE PATH, GET IT STRAIGHT FROM PARAMETER INSTEAD OF CALCULATING IT AGAIN
    input_file = input_file.get_text()
    print(input_file)
    # INPUT FILE PRE-PROCESSING FOR STRING SEARCH
    # INCLUDES TRANSFORMATION OF DOUBLE SPACES AND NEW LINES TO SINGLE SPACES + LOWERCASING

    input_file = re.sub("\s+", r"  ", input_file)
    input_file = re.sub("(\s+\-)", r" - ", input_file)
    input_file = re.sub("([a-zA-Z]+)([0-9]+)", r"\1 \2", input_file)
    input_file = re.sub("([0-9]+)([a-zA-Z]+)", r"\1 \2", input_file)
    input_file = re.sub("([()!,;\.\?\[\]\|])", r" \1 ", input_file)
    input_file = re.sub("\s+", " ", input_file)
    input_file = input_file.lower()
    
    with open(keyword_guide_path,'w') as keyword_guide_file:
        print("\n"+title+"\n"+'Keywords found by String Search'+"\n", file=keyword_guide_file)
        for keyword in keywords:
            kw_counter = string_search(input_file,0,keyword.lower())
            if (kw_counter != 0):
                print(keyword+":"+str(kw_counter), file=keyword_guide_file)

    doc = nlp(input_file)
    doc = reconstruct_hyphenated_words(doc)
    doc = reconstruct_noun_chunks(doc,keywords)
    tokens = [token.text for token in doc if not token.is_space if not token.is_punct if not token.text in stopwords.words()]
    #nlp.add_pipe("merge_noun_chunks") # NOT NEEDED WITH THE NEW LOGIC THAT PUTS TOKETHER KEYWORDS
    print(tokens)
    
    
    print("\nWith stop word removal","\nSize of original corpus:", len(doc), "\nSize of filtered corpus:",len(tokens), file=output_file)

    compute_stats(tokens,title, output_file, source_path) #source+title

    output_file.close()

def analyse_folder(source):
    path='data/'+file_input_path_general
    for filename in os.listdir(path):
        if os.path.isdir(path+'/'+filename):
            analyse_folder(filename+'/')
        else:
            file_name, file_extension = os.path.splitext(filename)
            process_document(file_name, source, keywords)
    


#####
#  MAIN 
#####

nlp = spacy.load('en_core_web_sm')
#nlp.add_pipe("merge_noun_chunks")
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
output_file.close()

#### PERFORM DESCRIPTIVE STATISTICS ON ALL DATA

folder = int(input("Choose which folder to analyze\n1 for Facebook-Sourced\n2 for Academic Articles\n3 for Guidelines\n"))
path = ''#'data/'
if(folder == 1): 
    path+='Facebook/Privacy/TargetCompanySourced'
    source='TargetCompanySourced'
elif(folder == 2):
    path+='Facebook/Privacy/Academic Articles Facebook'
    source='Academic Articles Facebook'
elif(folder == 3):
    path+='Guidelines'
    source='Guidelines'

for filename in os.listdir('data/'+path):#'data/'+folder):
    print(filename)
    #print("!"+source+"!")
    file_name, file_extension = os.path.splitext(filename)
    process_document(file_name, path, source, keywords)
    #print(path, source, keywords)

# IF WE NEED TO RECREATE THE JOINT GRAPH, USE THIS COMMAND TO SAVE IT 
#plt.savefig('output/JointGraph.png', bbox_inches='tight')

#################
## COMMENTS ON PROJECT PROGRESS

# SOLVE PROBLEM - KEYWORDS ARE NOT APPEARING

# A - CORPUS PRE-PROCESSING

# LEMMATIZATION, STEMMING - I THINK IT'S A GOOD IDEA BUT WE SHOULD CHECK THE KEYWORDS
# EXPANDING ABBREVIATIONS?
# TO DO: FIGURE OUT HOW TO DEAL WITH THE COMMAS ',' AND PUNCTUATION THAT ARE BEING SEEING AS PART OF A TOKEN
# CURRENTLY EVALUATING WORDS INSIDE NOUN CHUNKS ONLY AS THE NOUN CHUNK SET, SO MAYBE WE ARE MISSING KEYWORDS INSIDE OF NOUN CHUNKS?

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

# TO DO: PLOT ALL OF THE GRAPHS TOGETHER TOO BUT READABLY and with no stop words!!!! check
# TO DO: TITLE TO THE GRAPHS check
# TO DO: N GRAMS
# TO DO: MODIFY GRAPH SO THAT THE FULL WORDS CAN BE READ check

# MEETING: ASK WHAT KINDS OF INFO/DESCRIPTIVE STATISTICS WE WANT TO OBTAIN THAT WE DONT ALREADY HAVE
# COULD APPLY FREQUENCY CUT-OFF - IS IT WORTH IT? GIVEN THAT PRIVACY IS SAID ONCE MAYBE, I COULD SAY NO BUT TITLE MAYBE SHOULDNT COUNT, BUT ALSO HOW TO RULE OUT/MAKE SURE IT IS THE TITLE THAT IS BEING CUT OFF


# LOWERCASING? probably not needed
# NUMBERS TO WORDS? REMOVE NUMBERS? probably not needed
# EXPANDING ABBREVIATIONS? 
# READING OUT DATES? probably not needed
# TO DO: FIGURE OUT HOW TO DEAL WITH THE COMMAS ',' AND PUNCTUATION THAT ARE BEING SEEING AS PART OF A TOKEN!!!!
# TO DO: FIND A WAY TO CHECK MEMORY LEAKS - ASK PROF RICCARDI?
# AFTER ORGANIZING THIS ALL -> MOVE ON TO DEPENDENCY PARSING

# I realized I should sent an email warning you whenever I update the output in the folders, from now on I will, and where to find the changes

# PREPARE MEETING
# NON CODING TASKS - friday pre-meeting tasks
#
# READ FACEBOOK SOURCED FILES CAREFULLY
# EXAMINE 2 FILES WITH ESG APPROACHES ---- do not follow this angle yet
# Take a look at stats, come up with an opinion in new keywords --- do not follow this angle yet


# To debug f5, import gc gc.collect() but I'm not yet satidfied with the results

# BRAINSTORMING

# Should we have a DPIA to analyze as input??


### NEXT STEPS:

# SEARCH STRING OF ALL TERMS
# CLEANING
# FILTERING
# TOKENIZATION
# TESTING WITH SPACY DEPENDENCY PARSER
# DEPENDENCY PARSING
# VERSION WITH LEMMATIZATION TOO

# include plural forms! e.g. data breaches
# NIST 's () --- how to deal

#
# Apostrophe held specifically in english

#
# NEXT STEP: TAKE CARE OF WEIRDLY TOGETHER CASES "5 3", "1 THIRD"