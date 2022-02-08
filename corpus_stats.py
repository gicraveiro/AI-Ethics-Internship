import os
import pdfx
import spacy
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from utils import clean_corpus, reconstruct_hyphenated_words, reconstruct_noun_chunks

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
    graph_path='output/Corpus Statistics/'+gen_path+'/'+filename+'/Graph.png'
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    freq = plot_graph(tokens,graph_path,filename)

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
    print("----------------------------------------------------------------------------------------", file=output_file)
    print("\nKeywords that appear in the file, alongside with their frequencies:", file=output_file)
    for keyword in keywords_present:
        print(keyword, file=output_file)

def string_search(document, index,keyword):
    counter=0
    index = document.find(keyword, index+1) 
    if(index != -1): 
        counter +=1
        counter += string_search(document,index,keyword)
    return counter

def parser(corpus, output_file):
    for token in corpus:
        for keyword in keywords:
            if(token.text == keyword.lower()):
                print("\n\nKEYWORD:",token.text,"->", token.dep_, file=output_file)
                print("\nDescendants:", file=output_file)
                for descendant in token.subtree:
                    if(descendant != token and descendant.dep_ != "det" and descendant.dep_ != "punct" and descendant.dep_ != "prep" and descendant.dep_ != "aux" and descendant.dep_ != "auxpass"):
                        print(descendant.text, "->", descendant.dep_, file=output_file)
                print("\nAncestors:",file=output_file)
                for ancestor in token.ancestors:
                    if(ancestor != token and ancestor.dep_ != "det" and ancestor.dep_ != "punct" and ancestor.dep_ != "prep" and ancestor.dep_ != "aux" and ancestor.dep_ != "auxpass"):
                        print(ancestor.text, "->", ancestor.dep_, file=output_file)
                print("\nImmediate children:", file=output_file)
                for child in token.children:
                    if(child != token and child.dep_ != "det" and child.dep_ != "punct" and child.dep_ != "prep" and child.dep_ != "aux" and child.dep_ != "auxpass"):
                        print(child.text, "->", child.dep_, file=output_file)

def process_document(title, source_path,keywords):
    
    # CREATING OUTPUT FILES
    if(len(keywords) > 1):
        stats_path = 'output/Corpus Statistics/'+source_path+'/'+title+'/Stats.txt'
        keyword_guide_path = 'output/Corpus Statistics/'+source_path+'/'+title+'/KeywordsFound.txt'
    else:
        stats_path = 'output/Corpus Statistics/'+source_path+'/'+title+'/PrivacyOnlyStats.txt'
        keyword_guide_path = 'output/Corpus Statistics/'+source_path+'/'+title+'/PrivacyOnlyKeywordsFound.txt'
    
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    os.makedirs(os.path.dirname(keyword_guide_path), exist_ok=True)
    output_file = open(stats_path, 'w')
    print("\n"+title+"\n", file=output_file)
    
    # READING AND MANIPULATING INPUT FILE
    path = 'data/'+source_path+'/'+title+'.pdf'
    input_file = pdfx.PDFx(path) 
    input_file = input_file.get_text()

    # INPUT FILE PRE-PROCESSING FOR STRING SEARCH
    input_file = clean_corpus(input_file)
   
    with open(keyword_guide_path,'w') as keyword_guide_file:
        print("\n"+title+"\n"+'Keywords found by String Search'+"\n", file=keyword_guide_file)
        for keyword in keywords:
            kw_counter = string_search(input_file,0,keyword.lower())
            if (kw_counter != 0):
                print(keyword+":"+str(kw_counter), file=keyword_guide_file)

    doc = nlp(input_file)
    doc = reconstruct_hyphenated_words(doc)
    doc = reconstruct_noun_chunks(doc,keywords) # ONLY MERGES TO THE SAME TOKEN THE COMPUND KEYWORDS, so if the only keyword is privacy, it doesn't do anything
    tokens = [token for token in doc if not token.is_space if not token.is_punct if not token.text in stopwords.words()] # token for parser, token.text for frequency test
    
    print("\nWith stop word removal","\nSize of original corpus:", len(doc), "\nSize of filtered corpus:",len(tokens), file=output_file)

    compute_stats([token.text for token in tokens],title, output_file, source_path) 
    
    print("---------------------------------------------------------------------------------------", file=output_file)
    print("Dependency relations of keywords that appear in the file:", file=output_file)
    parser(tokens, output_file) # dependency relations

    output_file.close()

#################
#  MAIN 

nlp = spacy.load('en_core_web_lg') # original one was with small version
nlp.add_pipe("merge_entities")

kw_opt = int(input("Enter the preferred option:\nFor 'privacy' as the only keyword, enter 1\nFor the keywords list, enter 2\n"))


# PRIVACY AS ONLY KEYWORD
if(kw_opt == 1):
    keywords = ['privacy']

# KEYWORDS 
elif (kw_opt == 2):
    output_file = open('output/Corpus Statistics/Keywords.txt', 'w')
    keywords_file = open('data/Utils/PrivacyKeyWords2.txt', "r", encoding='utf-8-sig') 
    #keywords_file = open('data/Utils/PrivacyKeyWords.txt', "r", encoding='utf-8-sig') # Original list of keywords( without different spellings)
    keywords = keywords_file.read()
    keywords = keywords.split(", ")
    keywords_file.close()

    print("Descriptive Statistics of Facebook Sourced Files on Privacy", file=output_file)
    print("\nKeywords:\n",file=output_file)
    for keyword in keywords:
        print(keyword, file=output_file)
    output_file.close()
else:
    print("This was not a valid option, bye bye")
    quit()


#### PERFORM DESCRIPTIVE STATISTICS ON ALL DATA

folder = int(input("Choose which folder to analyze\n1 for Facebook Privacy Policies\n2 for Academic Articles related to Facebook and Privacy\n3 for AI Ethics Guidelines on Privacy\n"))
path = ''
if(folder == 1): 
    path+='Facebook/Policies'
elif(folder == 2):
    path+='Facebook/Academic Articles'
elif(folder == 3):
    path+='Guidelines'

# LOOP THROUGH FILES
for filename in os.listdir('data/'+path):
    print(filename)
    file_name, file_extension = os.path.splitext(filename)
    process_document(file_name, path, keywords)

# TO RECREATE THE GRAPH THAT CONTAINS WORD FREQUENCY FOR ALL DOCUMENTS, USE THIS COMMAND TO SAVE IT 
#plt.savefig('output/Corpus Statistics/JointGraph.png', bbox_inches='tight')

# Note: 
# Due to alignment in preprocessing and multiple corrections and improvements throughout the internship, original results are no longer reproducible...
