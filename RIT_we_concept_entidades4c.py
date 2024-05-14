import pandas as pd
import numpy as np
import utils as ut # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import spacy
import mutual_info as mi
import time
from scipy.stats import wasserstein_distance
import sys
from math import floor


import conceptnet_lite
conceptnet_lite.connect("../OPENAI/data/conceptnet.db")
from conceptnet_lite import Label, edges_for, edges_between

def obtener_distancia(texto_v,hipotesis_v,texto_t,texto_h,b_col,b_index):
    lista_l=[]
    for i in range(len(texto_t)):
        lista=[]
        for j in range(len(texto_h)):
            lista.append(np.linalg.norm(texto_v[i] - hipotesis_v[j]))#*wasserstein_distance(texto_2[i],hipotesis_2[j]))
        lista_l.append(lista)
    df_distEuc=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    df_distEuc=df_distEuc.drop(b_col[1:],axis=1)
    df_distEuc=df_distEuc.drop(b_index[1:],axis=0)
    return df_distEuc

def wasserstein_mutual_inf(texto_v,hipotesis_v,texto_t,texto_h):  
    lista_l=[]
    lista_muinfor=[]   
    for i in range(len(texto_t)):
        lista=[]
        lista_mu=[]
        for j in range(len(texto_h)):
            lista.append(wasserstein_distance(texto_v[i],hipotesis_v[j]))
            lista_mu.append(mi.mutual_information_2d(np.array(texto_v[i]),np.array(hipotesis_v[j])))
        lista_l.append(lista)
        lista_muinfor.append(lista_mu)
    DFmearth=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    DFmutual_inf=pd.DataFrame(lista_muinfor,index=texto_t,columns=texto_h)
    return DFmearth,DFmutual_inf

def entropia(X):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probs = [np.mean(X == valor) for valor in set(X)]
    return round(sum(-p * np.log2(p) for p in probs), 3)

relaciones_generales=["is_a","etymologically_related_to","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent"]
relaciones_especificas=["is_a","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent"]

def bag_of_synonyms(word):
    sinonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name == "synonym":
                if word== e.start.text:
                    sinonimos.add(e.end.text)
                elif word== e.end.text:
                    sinonimos.add(e.start.text)
    except:
        pass
    sinonimos.add(word)
    return sinonimos

def bag_of_antonyms(word):
    antonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in ["antonym","distinc_from"]:
                if word== e.start.text:
                    antonimos.add(e.end.text)
                elif word== e.end.text:
                    antonimos.add(e.start.text)
    except:
        pass
    return antonimos

def bag_of_hyperonyms(word):
    hiperonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in relaciones_generales:
                if word== e.start.text:
                    hiperonimos.add(e.end.text)
    except:
        pass
    return hiperonimos

def bag_of_hyponyms(word):
    hiponimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in relaciones_especificas:
                if word== e.end.text:
                    hiponimos.add(e.start.text)
                    #print(e.relation.name,e.start.text)
    except:
        pass
    return hiponimos

def jaro_distance(s1, s2,sinT,sinH,HipT,hipH) :
    bandera=True

    # Length of two strings
    len1 = len(s1)
    len2 = len(s2)

    # If the listas de tokens are equal 
    if len1==len2:
        for i in range(len1):
            if s1[i]!=s2[i]:
                bandera=False
                break
        if (bandera):
            return 1.0; 
 
    if (len1 == 0 or len2 == 0) :
        return 0.0; 
 
    # Maximum distance upto which matching 
    # is allowed 
    max_dist = (max(len(s1), len(s2)) // 2 )-1 ; 
 
    # Count of matches 
    match = 0; 
 
    # Hash for matches 
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string 
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist), 
                       min(len2, i + max_dist + 1)):
            #print(s1[i],s2[j])
            # If there is a match or is contain in a bag of sinomys of tk
            if ((s1[i] == s2[j] or s1[i] in sinH[j] or s2[j] in sinT[i]) and hash_s2[j] == 0) : 
                print(s1[i],s2[j],"sinonimos")
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif ((s1[i] in hipH[j] or len((sinT[i]).intersection(hipH[j]))>0) and hash_s2[j] == 0):
                print("hiponimos",s2[j],s1[i],(sinT[i]).intersection(hipH[j]))
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif ((s2[j] in HipT[i] or len((sinH[j]).intersection(HipT[i]))>0) and hash_s2[j] == 0): 
                print("hiperonimos sobre sinonimos",s2[j],s1[i])
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif len((hipH[j]).intersection(HipT[i]))>0 and hash_s2[j] == 0: 
                print("hiperonimos3",s2[j],s1[i],(hipH[j]).intersection(HipT[i]))
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            
    print(hash_s1)
    print(hash_s2)
    print(match)
    # If there is no match 
    if (match == 0) :
        return 0.0; 
 
    # Number of transpositions 
    t = 0; 
 
    point = 0; 
 
    # Count number of occurrences 
    # where two characters match but 
    # there is a third matched character 
    # in between the indices 
    for i in range(len1) : 
        if (hash_s1[i]) :
            # Find the next matched character 
            # in second string 
            while (hash_s2[point] == 0) :
                point += 1; 
 
            if (s1[i] != s2[point]) :
                point += 1
                t += 1
            else :
                point += 1    
    t /= 2; 
    #Return the Jaro Similarity 
    return match / len2; 

def relacion_entailment(wt,wh):
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        for e in edges_between(concepts_wt, concepts_wh):
            if wt == e.start.text and e.relation.name in relaciones_generales:
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
    except:
        pass
    return False

def relacion_noentailment(wt,wh):
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        for e in edges_between(concepts_wt, concepts_wh):
            if e.relation.name in ["distinct_from","antonym"]:
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
    except:
        pass
    return False

def relacion_conceptual(wt,wh):
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        for e in edges_between(concepts_wt, concepts_wh):
            if e.relation.name in ["related_to","similar_to"]:
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
    except:
        pass
    return False

def negacion(nlp,texto):
    b=1.0
    if (type(texto)==type(b) or texto=="" or texto=="n/a" or texto=="nan"):
        return 0,""
    doc = nlp(texto.lower())
    for token in doc:
        if(token.dep_=="neg"):
            return 1, token.head.lemma_
    return 0,""

def representacion_entidades(nlp,texto):
    doc = nlp(texto.lower())
    dir_sust=dict()
    palabras=[]
    for token in doc:
        if token.dep_ in tags:
            #print(token.text, token.lemma_, token.pos_,token.dep_,token.head.text,token.head.lemma_, token.head.pos_,
            #    [child for child in token.children])
            #sustantivos.append(token.head.lemma_)
            if token.head.lemma_ in dir_sust:
                if dir_sust[token.head.lemma_]=="NA":
                    dir_sust[token.head.lemma_]=token.lemma_
                    if token.head not in palabras:
                        palabras.append(token.head)
                else:
                    dir_sust[token.head.lemma_]=dir_sust[token.head.lemma_]+","+token.lemma_
                    if token.head not in palabras:
                        palabras.append(token.head)
            else:
                dir_sust[token.head.lemma_]=token.lemma_
                if token.head not in palabras:
                    palabras.append(token.head)
        elif token.pos_ in ["NOUN","PROPN"]:
            #print(token.text,token.lemma_, token.pos_,token.dep_)
            #sustantivos.append(token.lemma_)
            if token.lemma_ not in dir_sust:
                dir_sust[token.lemma_]="NA"
                if token not in palabras:
                    palabras.append(token)
        elif token.pos_ in ["VERB"]:
            #print(token.text, token.lemma_,token.pos_,token.dep_)
            #sustantivos.append(token.lemma_)
            if token.lemma_ not in dir_sust:
                dir_sust[token.lemma_]="NA"
                if token not in palabras:
                    palabras.append(token)
    return dir_sust,palabras

def representacion2(nlp,texto):
    doc = nlp(texto.lower())
    dir_sust=dict()
    palabras=[]
    noun_phrase= [chunk.lemma_ for chunk in doc.noun_chunks]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_!=chunk.lemma_:
            dir_sust[chunk.root.lemma_]=','.join(chunk.lemma_.split()[:-1])
        else:
            dir_sust[chunk.root.lemma_]="NA"
    palabras.extend([chunk.root.lemma_ for chunk in doc.noun_chunks])
    palabras.extend(verbs)
    return dir_sust,palabras

deps_tags=["relcl","prep","amod","compound","pobj","conj","nsubj","advcl","dobj","neg",'acomp',"attr", 'nummod',"oprd","npadvmod"]
palabras_no=["wear","dress","clothe"]
def representacion(nlp,texto):
    dir_sust=dict()
    palabras=[]
    b=1.0
    if (type(texto)==type(b) or texto=="" or texto=="n/a" or texto=="nan"):
        return dir_sust,palabras
    doc =nlp(texto)
    doc =nlp(texto)
    # poses=[]
    # tokens=[]
    # lemmas=[]
    # children=[]
    # verb_vincu=[]
    # deps=[]
    dir_sust=dict()
    palabras=[]
    frase=""
    anterior="det"
    nsubj=""
    for token in doc:
        #print(token.text, token.dep_, token.head.text, token.head.pos_,
        #    [child for child in token.children])
        # poses.append(token.pos_)
        # tokens.append(token.text)
        # lemmas.append(token.lemma_)
        # verb_vincu.append(token.head.lemma_)
        # deps.append(token.dep_)
        # children.append([child for child in token.children])
        # print(token.text,token.pos_,token.dep_)
        if token.lemma_ not in palabras_no:
            if token.dep_ in deps_tags or (token.dep_=="ROOT" and token.pos_=="NOUN"):
                if token.dep_=="nsubj": 
                    if nsubj=="":
                        frase= frase + " " + token.lemma_ 
                        nsubj=token.lemma_
                    else:
                        if frase!="":
                            palabras.append(frase.split()[-1])
                            if len(frase.split())>1:
                                dir_sust[frase.split()[-1]]=','.join(frase.split()[0:-1])
                            else:
                                dir_sust[frase.split()[-1]]="NA"
                        frase=""
                elif token.dep_=="prep":
                    if token.lemma_ not in ["in","at","on","with","from","to","for","since"]:    
                        frase= frase + token.lemma_ + " "
                    else:
                        if frase!="":
                            palabras.append(frase.split()[-1])
                            if len(frase.split())>1:
                                dir_sust[frase.split()[-1]]=','.join(frase.split()[0:-1])
                            else:
                                dir_sust[frase.split()[-1]]="NA"
                        frase=""
                elif token.dep_=="neg":
                    frase= frase + " "+ token.lemma_
                else:
                    
                    frase= frase + " "+ token.lemma_
            elif token.dep_ in ["ROOT"] and token.pos_ in ["VERB"] and anterior=="NUM":
                #palabras.append(frase+" "+token.lemma_)
                palabras.append(token.lemma_)
                frase=""
            elif token.dep_ in ["ROOT"] and token.pos_ in ["VERB"]:
                if frase!="":
                    palabras.append(frase.split()[-1])
                    if len(frase.split())>1:
                        dir_sust[frase.split()[-1]]=','.join(frase.split()[0:-1])
                    else:
                        dir_sust[frase.split()[-1]]="NA"
                    palabras.append(token.lemma_)
                    dir_sust[token.lemma_]="NA"
                else:
                    palabras.append(token.lemma_)
                    dir_sust[token.lemma_]="NA"
                frase=""
            else:
                #print(frase)
                if frase!="":
                    palabras.append(frase.split()[-1])
                    if len(frase.split())>1:
                        dir_sust[frase.split()[-1]]=','.join(frase.split()[0:-1])
                    else:
                        dir_sust[frase.split()[-1]]="NA"
                frase=""
        anterior=token.pos_
    if frase!="":
        palabras.append(frase.split()[-1])
        if len(frase.split())>1:
            dir_sust[frase.split()[-1]]=','.join(frase.split()[0:-1])
        else:
            dir_sust[frase.split()[-1]]="NA"
    # poses.append("<F>")
    # tokens.append("</F>")
    # lemmas.append("</F>")
    # children.append("</F>")
    # verb_vincu.append("</F>")
    # deps.append("</F>")
    # print("tokens",tokens)
    # print("poses",poses)
    # print("lemmas",lemmas)#
    # print("children",children)
    # print("verbos_vinculantes",verb_vincu)#
    # print("deps",deps)
    return dir_sust,palabras



nlp = spacy.load("en_core_web_md") # modelo de nlp

tags=[ 'acl',
 'acomp',
 'advcl',
 'advmod',
 'amod',
 'appos',
 'ccomp',
 'complm',
 'compound',
 'conj',
 'infmod',
 'meta',
 'neg',
 'nmod',
 'nn',
 'npadvmod',
 'nounmod',
 'npmod',
 'num',
 'number',
 'nummod',
 'partmod',
 'pcomp',
 'poss',
 'possessive',
 'prep',
 'quantmod',
 'rcmod',
 'relcl',
 'xcomp',
 'adc',
 'avc',
 'mnr',
 'mo',
 'ng',
 'nmc'
]

#ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv
ut.load_vectors_in_lang(nlp,"./data/numberbatch-en-17.04b.txt") # carga de vectores en nlp.wv

#prueba=pd.read_csv("data/DEV/pruebaDEV.csv")
prueba=pd.read_csv("../OPENAI/data/"+sys.argv[1])

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()

# lista de listas para dataframe
new_data = {'sumas' : [], 'distancias' : [], 'entropia_total' : [],'entropias' : [],'mutinf' : [], 
            'mearts' : [], 'max_info' : [],  'list_comp' : [], 'diferencias' :[], 'list_incomp':[],
            'list_M' : [], 'list_m' : [], 'list_T' : [], 'Jaro-Winkler_rit':[],'rel_conceptuales':[],
            'negT' : [], 'verbT' : [], 'negH' : [], 'verbH':[], 'overlap_ent':[],'clases' : []}


# cargar los conjuntos de sinonimos, hyperonimos e hyponimos
diccionario_sinonimos=dict()
diccionario_hiperonimos=dict()
diccionario_hyponimos=dict()
#diccionario_antonimos=dict()

df_temp=pd.read_pickle("salida/nuevo4a/Synonyms.pickle")
for index,strings in df_temp.iterrows():
    diccionario_sinonimos[strings['word']]=strings['Synonym']

df_temp=pd.read_pickle("salida/nuevo4a/Hyperonyms.pickle")
for index,strings in df_temp.iterrows():
    diccionario_hiperonimos[strings['word']]=strings['Hyperonym']

df_temp=pd.read_pickle("salida/nuevo4a/Hyponyms.pickle")
for index,strings in df_temp.iterrows():
    diccionario_hyponimos[strings['word']]=strings['Hyponym']

inicio = time.time()
for i in range(len(textos)):
    print(i)
    print(textos[i])
    # #r_t,t_clean_m=representacion_entidades(nlp,textos[i])
    # #r_t,t_clean_m=representacion2(nlp,textos[i])
    r_t,t_clean_m=representacion(nlp,textos[i])
    neg_t,negadat=negacion(nlp,textos[i])
    new_data['negT'].append(neg_t)
    new_data['verbT'].append(negadat)
    # #print(list(r_t.keys()))
    # for clave in r_t.keys():
    #     print("texto",clave,r_t[clave])
    print(hipotesis[i])
    # #r_h,h_clean_m = representacion_entidades(nlp,hipotesis[i])
    # #r_h,h_clean_m = representacion2(nlp,hipotesis[i])
    r_h,h_clean_m = representacion(nlp,hipotesis[i])
    neg_h,negadah=negacion(nlp,hipotesis[i])
    new_data['negH'].append(neg_h)
    new_data['verbH'].append(negadah)
    # for clave in r_h.keys():
    #     print("hipotesis",clave,r_h[clave])
    if len(set(h_clean_m))!=0 and len(set(t_clean_m))!=0:
        new_data['overlap_ent'].append(len(set(t_clean_m).intersection(set(h_clean_m)))/len(set(h_clean_m)))
    else:
        new_data['overlap_ent'].append(0)
    t_lem=ut.get_lemmas_(textos[i],nlp)
    h_lem=ut.get_lemmas_(hipotesis[i],nlp)
    t_vectors=ut.get_matrix_rep2(t_lem, nlp, normed=False)
    h_vectors=ut.get_matrix_rep2(h_lem, nlp, normed=False)
    t_vectors_n=ut.get_matrix_rep2(t_lem, nlp, normed=True)
    h_vectors_n=ut.get_matrix_rep2(h_lem, nlp, normed=True)
    
    # print(t_vectors)
    # print(h_vectors)
    # print(t_clean_m,h_clean_m)
    
    # s1=t_clean_m.split()
    # s2=h_clean_m.split()
    for t in t_lem:
        if t not in diccionario_sinonimos:
            diccionario_sinonimos[t]=bag_of_synonyms(t)
        if t not in diccionario_hiperonimos:
            diccionario_hiperonimos[t]=bag_of_hyperonyms(t)
        if t not in diccionario_hyponimos:
            diccionario_hyponimos[t]=bag_of_hyponyms(t)
        # if t not in diccionario_antonimos:
        #     diccionario_antonimos[t]=bag_of_antonyms(t)
    for t in h_lem:
        if t not in diccionario_sinonimos:
            diccionario_sinonimos[t]=bag_of_synonyms(t)
        if t not in diccionario_hiperonimos:
            diccionario_hiperonimos[t]=bag_of_hyperonyms(t)
        if t not in diccionario_hyponimos:
            diccionario_hyponimos[t]=bag_of_hyponyms(t)
        # if t not in diccionario_antonimos:
        #     diccionario_antonimos[t]=bag_of_antonyms(t)

    s1=t_lem
    s2=h_lem
    
    sinT=[]
    antT=[]
    HipT=[]
    sinH=[]
    antH=[]
    HipH=[]
    hipH=[]
    
    # # encontrar bolsa de sinonimos de cada token
    for t in s1:
        sinT.append(diccionario_sinonimos[t])
        HipT.append(diccionario_hiperonimos[t])

    for h in s2:
        sinH.append(diccionario_sinonimos[h])
        hipH.append(diccionario_hyponimos[h])

    print(t_lem)
    print(h_lem)
    tp1=jaro_distance(t_lem, h_lem,sinT,sinH,HipT,hipH)
    new_data['Jaro-Winkler_rit'].append(tp1)

    # # Obtencion de matriz de alineamiento, matriz de move earth y mutual information
    ma=np.dot(t_vectors_n,h_vectors_n.T)
    #print(t_clean,h_clean)
    #print(len(t_vectors_n),len(h_vectors_n),len(t_clean),len(h_clean))
    m_earth,m_mi=wasserstein_mutual_inf(t_vectors_n,h_vectors_n,t_lem,h_lem)
    ma=pd.DataFrame(ma,index=t_lem,columns=h_lem)
    #print(ma)

    # # Calculamos la entropia inicial de la matriz de distancias coseno sobre tokens de T y H
    new_data['entropia_total'].append(entropia(ma.round(1).values.flatten())) 

    # ###### BORRADO DE COSAS QUE NO OCUPO, SOLO NOS QUEDAMOS CON INFORMACIÓN DE TIPOS DE PALABRA: NOUN, VERB, ADJ Y ADV
    # # TAMBIÉN OMITIMOS EL VERBO BE DEBIDO A QUE POR LO REGULAR SE UTILIZA COMO AUXILIAR Y ES UN VERBO COPULATIVO
    # # sirve para construir la llamada predicación nominal del sujeto de una oración: 
    # # #el sujeto se une con este verbo a un complemento obligatorio llamado atributo que por lo general determina 
    # # alguna propiedad, estado o equivalencia del mismo, por ejemplo: "Este plato es bueno". "Juan está casado".
    c_compatibilidad=0
    c_incompatibilidad=0
    c_conceptuales=0
    # c_rel_concep=0
    b_col=[0]

    new_data['list_T'].append(ma.shape[0])
    new_data['list_M'].append(ma.shape[1])
    
    #procesamiento de cosas que son la misma entidad
    borrar=list(set(t_lem).intersection(set(h_lem)))
    ma = ma.drop(borrar,axis=1)
    m_earth = m_earth.drop(borrar,axis=1)
    m_mi = m_mi.drop(borrar,axis=1)
    ma = ma.drop(borrar)
    m_earth = m_earth.drop(borrar)
    m_mi = m_mi.drop(borrar)
    b_col.extend(borrar)

    # # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
    
    n_index = ma.shape[0]
    n_columns = ma.shape[1]
    pasada=0
    #print(ma.index,ma.columns)
    combinaciones=[]
    while n_columns>0 and n_index>0 and pasada<4:
        borrar=[]
        borrar_i=[]
        a = ma.idxmax().values
        b = ma.columns
        for j in range(len(a)):
            #print(a[j]+"_"+b[j] not in combinaciones,a[j]+"_"+b[j],combinaciones)
            if a[j]+"_"+b[j] not in combinaciones:
                combinaciones.append(a[j]+"_"+b[j])
                if(relacion_noentailment(a[j],b[j])):
                    borrar.append(b[j])
                    borrar_i.append(a[j])
                    c_incompatibilidad+=1
                elif(relacion_entailment(a[j],b[j])):
                    borrar.append(b[j])
                    borrar_i.append(a[j])
                    c_compatibilidad+=1
                else:
                    print("Proceso de conjuntos")
                    #sin1=bag_of_synonyms(a[j])
                    #sin2=bag_of_synonyms(b[j])
                    sin1=diccionario_sinonimos[a[j]]
                    sin2=diccionario_sinonimos[b[j]]
                    if len(sin1.intersection(sin2))>0:
                        borrar.append(b[j])
                        borrar_i.append(a[j])
                        c_compatibilidad+=1
                    else:
                        Hip1=set()
                        for e in list(sin1):
                            if e in diccionario_hiperonimos:
                                Hip1=Hip1.union(diccionario_hiperonimos[e])
                            else:
                                b_H=bag_of_hyperonyms(e)
                                Hip1=Hip1.union(b_H)
                                diccionario_hiperonimos[e]=b_H
                        if len(Hip1.intersection(sin2))>0:
                            borrar.append(b[j])
                            borrar_i.append(a[j])
                            c_compatibilidad+=1
                        else:
                            hip2=set()
                            for e in list(sin2):
                                if e in diccionario_hyponimos:
                                    hip2=hip2.union(diccionario_hyponimos[e])
                                else:
                                    b_H=bag_of_hyponyms(e)
                                    hip2=hip2.union(b_H)
                                    diccionario_hyponimos[e]=b_H                                
                            if len(sin1.intersection(hip2))>0:   
                                borrar.append(b[j])
                                borrar_i.append(a[j])
                                c_compatibilidad+=1
                            else:
                                if(relacion_conceptual(a[j],b[j])):
                                    borrar.append(b[j])
                                    borrar_i.append(a[j])
                                    c_conceptuales+=1
        pasada+=1
        ma = ma.drop(borrar,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        
        ma = ma.drop(borrar_i)
        m_earth = m_earth.drop(borrar_i)
        m_mi = m_mi.drop(borrar_i)
        n_index = ma.shape[0]
        n_columns = ma.shape[1]
        b_col.extend(borrar)
    b_index=[0]
    
    # #   ALMACENAMIENTO DE TODA LA INFORMACIÓN PROCESADA DE CARACTERÍSTICAS
    m_distancia = obtener_distancia(t_vectors,h_vectors,t_lem,h_lem,b_col,b_index)
    
    new_data['distancias'].append(m_distancia.max().sum()) #cambie de maximas a sumas
    m_earth=m_earth*m_distancia
    if ma.shape[1]==0 or ma.shape[0]==0:
        new_data['entropias'].append(0)
        new_data['max_info'].append(0)
        new_data['sumas'].append(0)
        new_data['mearts'].append(0)
        new_data['mutinf'].append(0)
        new_data['diferencias'].append(0)
    else:
        new_data['entropias'].append(entropia(ma.round(1).values.flatten()))
        new_data['max_info'].append(ma.max().sum()/(ma.shape[1]))# 
        new_data['sumas'].append(ma.sum().sum()/(ma.shape[1]))# 
        new_data['mearts'].append(m_earth.min().sum()/(ma.shape[1]))# 
        new_data['mutinf'].append(m_mi.max().sum()/(ma.shape[1]))# 
        new_data['diferencias'].append(len(ma.columns)/len(ma.index))

    new_data['list_comp'].append(c_compatibilidad)
    new_data['list_incomp'].append(c_incompatibilidad)
    new_data['rel_conceptuales'].append(c_conceptuales)
    new_data['list_m'].append(ma.shape[1])
    new_data['clases'].append(prueba.at[i,"gold_label"])
    print(ma)
fin = time.time()
df_resultados = pd.DataFrame(new_data)
df_resultados.to_pickle("salida/nuevo4c/"+sys.argv[1]+"_.pickle")
df = pd.DataFrame([[key, diccionario_sinonimos[key]] for key in diccionario_sinonimos.keys()], columns=['word', 'Synonym'])
# df.to_pickle("salida/nuevo4b/"+sys.argv[1]+"_Synonym.pickle")
# df = pd.DataFrame([[key, diccionario_hiperonimos[key]] for key in diccionario_hiperonimos.keys()], columns=['word', 'Hyperonym'])
# df.to_pickle("salida/nuevo4b/"+sys.argv[1]+"_Hyperonym.pickle")
# df = pd.DataFrame([[key, diccionario_hyponimos[key]] for key in diccionario_hyponimos.keys()], columns=['word', 'Hyponym'])
# df.to_pickle("salida/nuevo4b/"+sys.argv[1]+"_Hyponym.pickle")
# df = pd.DataFrame([[key, diccionario_antonimos[key]] for key in diccionario_antonimos.keys()], columns=['word', 'Antonym'])
# df.to_pickle("salida/nuevo4b/"+sys.argv[1]+"_Antonym.pickle")
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")