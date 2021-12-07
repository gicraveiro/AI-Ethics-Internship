import re

# reconstructs hyphen, slash and apostrophes
def reconstruct_hyphenated_words(corpus):
    i = 0
    while i < len(corpus):
        if((corpus[i].text == "-" or corpus[i].text == "/") and corpus[i].whitespace_ == ""): # identify hyphen ("-" inside a word)
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+2]) # merge the first part of the word, the hyphen and the second part of the word            
        elif(corpus[i].text == "’s" and corpus[i-1].whitespace_ == ""):
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+1])           
        else: 
            i += 1
    return corpus

# noun chunks that correspond to keywords
def reconstruct_noun_chunks(corpus,keywords):
    i = 0
    while i < len(corpus):
        counter = i
        token = corpus[i].text
        for keyword in keywords:
            kw_lower = keyword.lower()
            index = kw_lower.find(token)
            aux = index
            while (aux != -1 and counter < len(corpus)-1 and token != kw_lower):
                counter += 1
                token += ' '+corpus[counter].text
                aux = kw_lower.find(token)
                if(aux == -1):
                    counter -=1
                    token = corpus[i].text
            if(i != counter):
                if(token == kw_lower): 
                    with corpus.retokenize() as retokenizer:
                        retokenizer.merge(corpus[i:counter+1])
                    break 
                else: 
                    counter = i               
        if(i == counter):
            i += 1
    return corpus

def clean_corpus(corpus):
    corpus = corpus.lower()

    corpus = re.sub(" +", " ", corpus)
    corpus = re.sub("(\s+\-)", r" - ", corpus)
    corpus = re.sub("([a-zA-Z]+)([0-9]+)", r"\1 \2", corpus)
    corpus = re.sub("([0-9]+)([a-zA-Z]+)", r"\1 \2", corpus)
    corpus = re.sub("([()!,;\.\?\[\]\|])", r" \1 ", corpus)
    
    return corpus
