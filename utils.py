import re, os
import numpy as np

# Load vectors from dict
def load_vectors_as_dict(path):
    vectors = {}
    with open(path, 'r', encoding="utf8") as f:
        line = f.readline()
        while line:
            # Split on white spaces
            line = line.strip().split(' ')
            if len(line) > 2:
                vectors[line[0]] = np.array([float(l) for l in line[1:]], dtype=np.float32)
            line = f.readline()
    return vectors

# Load vectors in a spacy nlp
def load_vectors_in_lang(nlp, vectors_loc):
    wv= load_vectors_as_dict(vectors_loc)
    nlp.wv = wv

    # # Check if list of oov vectors exists
    # # If so, load, if not, create
    # oov_path,ext = os.path.splitext(vectors_loc)
    # oov_path = oov_path+'.oov.txt'
    # if os.path.exists(oov_path):
    #     nlp.oov = np.loadtxt(oov_path)
    # else:
    fk = list(wv.keys())[0]
    nf = wv[fk].shape[0]
    nlp.oov = np.random.normal(size=(100,nf))
    return 

# Get vector representation of word
def get_vector(w, nlp, nf=300):
    v = w.vector
    return v.astype(np.float32)

def get_vector2(w, nlp, nf=300):
    if str(w) in nlp.wv:
        v = nlp.wv[str(w)]
    else: 
        v = np.zeros((1,300))[0]
    return v.astype(np.float32)

# Some cleaning especially with respect to weird punctuation
def clean_text(s):
    s = re.sub("([.,!?()-])", r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

# Utility function to get a GloVe representation of a text
def get_matrix_rep(text, nlp, pos_to_remove=['PUNCT'], normed=True,
    lemmatize=False):

    text = clean_text(str(text)).lower()
#    text = text

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Lemmatize if desired
    if lemmatize:
        text = ' '.join([w.lemma_ for w in doc])
        doc = nlp(text)

    # Get processed words removing undesired POS
    words = [w for w in doc if w.pos_ not in pos_to_remove]
    #print("lo que se obtiene: ",words)
    # Get all vectors
    vecs = np.array([get_vector(w, nlp) for w in words], dtype=np.float32)
    if len(vecs) == 0:
        vecs = np.zeros((1,300), dtype=np.float32)

    # Normalize vectors if desired
    if normed:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
    return vecs

def get_matrix_rep2(words,nlp, normed=True):
    vecs = np.array([get_vector2(w,nlp) for w in words], dtype=np.float32)
    if len(vecs) == 0:
        vecs = np.zeros((1,300), dtype=np.float32)

    # Normalize vectors if desired
    if normed:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
    return vecs

def get_words_rep(text, nlp, pos_to_remove=['PUNCT'], normed=True,
    lemmatize=False):

    text = clean_text(str(text)).lower()    

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Lemmatize if desired
    if lemmatize:
        text = ' '.join([w.lemma_ for w in doc])
        doc = nlp(text)

    # Get processed words removing undesired POS
    words=[]
    for w in doc:
        if w.pos_ not in pos_to_remove:
            words.append(w.text+"{"+w.lemma_+","+w.pos_+"}")

    return words

def get_lemmas_(text, nlp, lemmatize=True):

    text = clean_text(str(text)).lower()    

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Get processed words removing undesired POS
    words=[]
    for token in doc:
        if token.pos_ in ["NUM","PROPN","NOUN","VERB","ADJ","ADV"] or token.dep_ =="neg":
            if token.lemma_ !="be":
                words.append(token.lemma_)
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #        token.shape_, token.is_alpha, token.is_stop)
        
    return words
def get_lemmas5_(text, nlp, lemmatize=True):

    text = clean_text(str(text)).lower()    

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Get processed words removing undesired POS
    words=[]
    for token in doc:
        if token.pos_ in ["NUM","PROPN","NOUN","VERB","ADJ","ADV","PRON","ADP","SCONJ"] or token.dep_ =="neg":
            if token.lemma_ !="be":
                words.append(token.lemma_)
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #        token.shape_, token.is_alpha, token.is_stop)
        
    return words
def get_words(text, nlp, pos_to_remove=['PUNCT'], normed=True,
    lemmatize=False):

    text = clean_text(str(text)).lower()    

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Lemmatize if desired
    if lemmatize:
        text = ' '.join([w.lemma_ for w in doc])
        doc = nlp(text)

    # Get processed words removing undesired POS
    words=[]
    for w in doc:
        if w.pos_ not in pos_to_remove:
            words.append(w.text)

    return words

def reform_sentence(frase,nlp):
    frase_m=[]
    b=1
    for i in range(len(frase)-1):
        word_h=frase[i]+"_"+frase[i+1]
        try:
            if word_h in nlp.wv:
                frase_m.append(frase[i])
                frase_m.append(word_h)
                b=0
            elif b==0:
                b=1
            else:
                frase_m.append(frase[i])
        except:
            pass
    text = " ".join(frase_m) + " " + frase[i+1]
    textoF=""
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ["NUM","PART","PROPN","NOUN","VERB","ADJ","ADV"]:
            textoF = textoF + token.text + " "
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #        token.shape_, token.is_alpha, token.is_stop)
    return textoF

def reform_sentence2(frase,nlp):
    frase.append("")
    frase_m=[]
    b=1
    for i in range(len(frase)-2):
        word_h=frase[i]+"_"+frase[i+1]
        try:
            #print(word_h)
            if word_h in nlp.wv:
                frase_m.append(frase[i])
                frase_m.append(word_h)
                b=0
            elif b==0:
                b=1
            else:
                frase_m.append(frase[i])
        except:
            pass
        word_h=frase[i]+"_"+frase[i+2]
        try:
            #print(word_h)
            if word_h in nlp.wv:
                frase_m.append(word_h)
                b=0
            elif b==0:
                b=1
        except:
            pass
        
    #return " ".join(frase_m) + " " + frase[i+1] + " " + frase[i+2]
    text = " ".join(frase_m) + " " + frase[i+1]
    return text


# Pad an array up to maxlen
def pad(X, maxlen):
    """Pads with 0 or truncates a numpy array along axis 0 up to maxlen
    Args:
        X (ndarray): array to be padded or truncated
        maxlen (int): maximum length of the array
    Returns:
        ndarray: padded or truncated array
    """

    nrows = X.shape[0]
    delta = maxlen - nrows
    if delta > 0:
        padding = ((0,delta), (0,0))
        return np.pad(X, pad_width=padding, mode='constant')
    elif delta < 0:
        return X[:maxlen,:]
    else:
        return X

def entropia(X):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probs = [np.mean(X == valor) for valor in set(X)]
    return round(np.sum(-p * np.log2(p) for p in probs), 3)

