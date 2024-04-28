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


import textacy
from textacy import extract
from functools import partial


## esta función revisa hiperonimia, sinonimia entre otras.
def compatibilidad_semantica(word_t,word_h):
    #Nos quedamos con el lemma para identificar las relaciones blue{ blue ,ADJ}​
    wt=str(word_t).split("{")[1].split(",")[0]
    wh=str(word_h).split("{")[1].split(",")[0]
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        #Sinonimia
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="synonym"):
            print(wh," synonym ",wt)
            return True,1,"synonym"
        #Antonimos
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="antonym"):
            print(wt," antonym ",wh)
            return True,2,"antonym"
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="distinct_from"):
            print(wh," distinct from ",wt)
            return True,2,"distinct_from"
        #Hiperonimia o Hiponimia
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="is_a"):
            if wt== e.start.text:
                print(wt," is_a ",wh)
                return True,1,"is_a"
            else:
                print(wt," is_a (ant) ",wh)
                return True,2,"is_a_c"
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="manner_of"):
            if wt== e.start.text:
                print(wt," manner_of ",wh)
                return True,1,"manner_of"
            else:
                print(wt," manner_of (ant) ",wh)
                return True,2,"manner_of_c"
        for e in edges_between(concepts_wh,concepts_wt, two_way=True,relation="has_a"):
            if wt== e.start.text:
                print(wh," has_a ",wt)
                return True,1,"has_a"
            else:
                print(wh,"has_a (ant)",wt)
                return True,2,"has_a_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="derived_from"):
            if wt== e.start.text:
                print(wt," derivado_from ",wh)
                return True,1,"derived_from"
            else:
                print(wh," derivado_from (ant)",wt)
                return True,2,"derived_from_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="used_for"):
            if wt== e.start.text:
                print(wt," used_for ",wh)
                return True,1,"used_for"
            else:
                print(wh,"used_for (ant)",wt)
                return True,2,"used_for_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_property"):
            if wt== e.start.text:
                print(wt," HasProperty ",wh)
                return True,1,"HasProperty"
            else:
                print(wh," HasProperty ",wt)
                return True,2,"hasProperty_c"
        # Inflexión
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="form_of"):
            print(wt," form_of ",wh)
            return True,1,"form_of"
        #Implicación
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="causes"):
            if wt== e.start.text:
                print(wt," causes ",wh)
                return True,1,"Causes"
            else:
                print(wh," causes ",wt)
                return True,2,"Causes_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_prerequisite"):
            if wt== e.start.text:
                print(wt," has_prerequisite ",wh)
                return True,1,"has_prerequisite"
            else:
                print(wh," has_prerequisite ",wt)
                return True,2,"has_prerequisite_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_subevent"):
            if wt== e.start.text:
                print(wt," has_subevent ",wh)
                return True,1,"has_subevent"
            else:
                print(wh," has_subevent ",wt)
                return True,2,"has_subevent_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_first_subevent"):
            if wt== e.start.text:
                print(wt," has_first_subevent ",wh)
                return True,1,"has_first_subevent"
            else:
                print(wh," has_first_subevent ",wt)
                return True,2,"has_first_subevent_c"
            
        #relaciones conceptuales
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="related_to"):
            print(wt," related_to ",wh)
            return True,3,"related_to"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="similar_to"):
            print(wt," similar_to ",wh)
            return True,3,"similar_to"
            
    except:
        pass
    return False,0,""

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

def bi_grams(texto1,texto2):
    gramsT=set()
    gramsH=set()
    palabras = texto1
    for i in range(len(palabras)-1):
        gramsT.add((palabras[i],palabras[i+1]))
    palabras = texto2
    for i in range(len(palabras)-1):
        gramsH.add((palabras[i],palabras[i+1]))
    if len(gramsH)!=0:
        return len(gramsT.intersection(gramsH))/len(gramsH)
    else:
        return 0

def tri_grams(texto1,texto2):
    gramsT=set()
    gramsH=set()
    palabras = texto1
    for i in range(len(palabras)-2):
        gramsT.add((palabras[i],palabras[i+1],palabras[i+2]))
    palabras = texto2
    for i in range(len(palabras)-2):
        gramsH.add((palabras[i],palabras[i+1],palabras[i+2]))
    if len(gramsH)!=0:
        return len(gramsT.intersection(gramsH))/len(gramsH)
    else:
        return 0

def cuatri_grams(texto1,texto2):
    gramsT=set()
    gramsH=set()
    palabras = texto1
    for i in range(len(palabras)-3):
        gramsT.add((palabras[i],palabras[i+1],palabras[i+2],palabras[i+3]))
    palabras = texto2
    for i in range(len(palabras)-3):
        gramsH.add((palabras[i],palabras[i+1],palabras[i+2],palabras[i+3]))
    if len(gramsH)!=0:
        return len(gramsT.intersection(gramsH))/len(gramsH)
    else:
        return 0

def get_grams_entities(texto):
    finales=[]
    doc = textacy.make_spacy_doc(texto, lang="en_core_web_md")
    terms = list(extract.terms(doc,
    ngs=partial(extract.ngrams, n=2, include_pos={"NOUN", "ADJ", "VERB","ADV","PART"}),
    ents=partial(extract.entities, include_types={"PERSON", "ORG", "GPE", "LOC"}),
    dedupe=False))
    a = list(extract.subject_verb_object_triples(doc))
    for e in terms:
        finales.append(str(e))
    for e in a:
        trip = ""
        for e1 in e[0]:
            trip += str(e1) + " "
        for e1 in e[1]:
            trip += str(e1) + " "
        for e1 in e[2]:
            trip += str(e1) + " "
        finales.append(trip)
    
    return finales

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

relaciones=["is_a","etymologically_related_to","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent"]

def bag_of_hyperonyms(word):
    hiperonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in relaciones:
                if word== e.start.text:
                    hiperonimos.add(e.end.text)
    except:
        pass
    return hiperonimos

def bag_of_hyponyms(word):
    hiponimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in relaciones:
                if word== e.end.text:
                    hiponimos.add(e.start.text)
    except:
        pass
    return hiponimos

def jaro_distance(s1, s2,sinT,sinH,HipT,hipH) :
    # sinT=[]
    # sinH=[]
    # HipT=[]
    # hipH=[]

    # for t in s1:
    #     sinT.append(bag_of_synonyms(t))
    #     HipT.append(bag_of_hyperonyms(t))
    # for h in s2:
    #     sinH.append(bag_of_synonyms(h))
    #     hipH.append(bag_of_hyponyms(h))
    # print("sinonimos de T",sinT)
    # print("Hiperonimos de T",HipT)
    # print("sinonimos de H",sinH)
    # print("hiponimos de h",hipH)


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
            return 1.0,0.0,0.0; 
 
    if (len1 == 0 or len2 == 0) :
        return 0.0,0.0,0.0; 
 
    # Maximum distance upto which matching 
    # is allowed 
    max_dist = (max(len(s1), len(s2)) // 2 )-1 ; 
 
    # Count of matches 
    match = 0; 
 
    # Hash for matches 
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string 
    for i in range(len1) : 
            
        # Check if there is any matches 
        for j in range( max(0, i - max_dist),
                    min(len2, i + max_dist + 1)) : 
            #print(s1[i],s2[j])
            # If there is a match or is contain in a bag of sinomys of tk
            if ((s1[i] == s2[j] or s1[i] in sinH[j] or s2[j] in sinT[i]) and hash_s2[j] == 0) : 
                print(s1[i],s2[j],"sinonimos")
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif ((s1[i] in hipH[j] or len((sinT[i]).intersection(hipH[j]))>0) and hash_s2[j] == 0):
                print("hiperonimos",s2[j],s1[i],(sinT[i]).intersection(hipH[j]))
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
        return 0.0,0.0,0.0; 
 
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
    print(t)
    #Return the Jaro Similarity 
    return ((match / len2 +
            (match - t) / match ) / 2.0),t,(len2-match)/len2; 

def jaro_distance_relacionadas(s1, s2,sinT,HipT,sinH,hipH) :
    
    # for t in s1:
    #     sinT.append(bag_of_synonyms(t))
    #     HipT.append(bag_of_hyperonyms(t))
    # for h in s2:
    #     sinH.append(bag_of_synonyms(h))
    #     hipH.append(bag_of_hyponyms(h))
    # print("sinonimos de T",sinT)
    # print("Hiperonimos de T",HipT)
    # print("sinonimos de H",sinH)
    # print("hiponimos de h",hipH)

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
            return 1.0,0.0; 
 
    if (len1 == 0 or len2 == 0) :
        return 0.0,0.0; 
 
    # Maximum distance upto which matching 
    # is allowed 
    max_dist = (max(len(s1), len(s2)) // 2 ) -1; 
 
    # Count of matches 
    match = 0; 
 
    # Hash for matches 
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string 
    for i in range(len1) : 
            
        # Check if there is any matches 
        for j in range( max(0, i - max_dist),
                    min(len2, i + max_dist + 1)) : 
            #print(s1[i],s2[j])
            # If there is a match or is contain in a bag of sinomys of tk
            if ((s1[i] in hipH[j] or len((sinT[i]).intersection(hipH[j]))>0) and s1[i]!=s2[j] and hash_s2[j] == 0):
                print("hiperonimos",s2[j],s1[i],(sinT[i]).intersection(hipH[j]))
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif ((s2[j] in HipT[i] or len((sinH[j]).intersection(HipT[i]))>0) and s1[i]!=s2[j] and hash_s2[j] == 0): 
                print("hiperonimos sobre sinonimos",s2[j],s1[i])
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif len((hipH[j]).intersection(HipT[i]))>0 and s1[i]!=s2[j] and hash_s2[j] == 0: 
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
        return 0.0,0.0; 
 
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
    print(t)
    #Return the Jaro Similarity 
    return (( match / len2  +
            (match - t) / match ) / 2.0),t; 
    
def jaro_distance_contra(s1, s2,antT,HipT,antH,HipH) :
    # antT=[]
    # antH=[]
    # HipT=[]
    # HipH=[]

    # for t in s1:
    #     antT.append(bag_of_antonyms(t))
    #     HipT.append(bag_of_hyperonyms(t))
    # for h in s2:
    #     antH.append(bag_of_antonyms(h))
    #     HipH.append(bag_of_hyperonyms(h))
    # print("antonimos de T",antT)
    # print("antonimos de h",antH)
    # print("Hiperonimos de T",HipT)
    # print("Hiperonimos de H",HipH)

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
            return 1.0,0.0; 
 
    if (len1 == 0 or len2 == 0) :
        return 0.0,0.0; 
 
    # Maximum distance upto which matching 
    # is allowed 
    max_dist = (max(len(s1), len(s2)) // 2 ) -1 ; 
 
    # Count of matches 
    match = 0; 
 
    # Hash for matches 
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string 
    for i in range(len1) : 
            
        # Check if there is any matches 
        for j in range( max(0, i - max_dist),
                    min(len2, i + max_dist + 1)) : 
            #print(s1[i],s2[j])
            # If there is a match or is contain in a bag of sinomys of tk
            if ((s1[i] in antH[j] or s2[j] in antT[i]) and s1[i]!=s2[j] and hash_s2[j] == 0) : 
                print(s1[i],s2[j],"antonimos")
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif (len(HipH[j].intersection(HipT[i]))>0 and s1[i]!=s2[j] and hash_s2[j] == 0):
                print(s1[i],s2[j],HipH[j].intersection(HipT[i]),"antonimos")
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
    print(hash_s1)
    print(hash_s2)
    print(match)
    # If there is no match 
    if (match == 0) :
        return 0.0,0.0; 
 
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
    print(t)
    #Return the Jaro Similarity 
    return (( match / len2 +
            (match - t) / match ) / 2.0),t; 

nlp = spacy.load("en_core_web_md") # modelo de nlp

#ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv
ut.load_vectors_in_lang(nlp,"./data/numberbatch-en-17.04b.txt") # carga de vectores en nlp.wv

#prueba=pd.read_csv("data/DEV/pruebaDEV.csv")
prueba=pd.read_csv("../OPENAI/data/"+sys.argv[1])

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()

# lista de listas para dataframe
new_data = {'sumas' : [], 'distancias' : [], 'entropia_total' : [],'entropias' : [],'mutinf' : [], 
            'mearts' : [], 'max_info' : [], 'similitud_faltantes' : [], 'list_comp' : [],
            'list_incomp' : [], 'list_rel_con' : [], 'list_M' : [], 'list_m' : [],
            'list_T' : [], 'list_relaciones' : [], 'listas_malign' : [], 'listas_malignf' : [],
            'list_bigram' : [], 'diferencias':[],'bi_ent_trip_t':[],'bi_ent_trip_h':[],'bi_ent_trip_rel':[], 
            'Jaro-Winkler_rit':[], 'nomatch':[],'entidades':[],'clases' : []}

inicio = time.time()
for i in range(len(textos)):
#for i in range(5):
    print(i)

    t_clean_m=ut.get_words(textos[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
    h_clean_m=ut.get_words(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)

    t_clean_m=ut.reform_sentence2(t_clean_m,nlp)
    h_clean_m=ut.reform_sentence2(h_clean_m,nlp)

    t_clean=ut.get_words_rep(t_clean_m,nlp,pos_to_remove=[""],normed=True,lemmatize=False)
    h_clean=ut.get_words_rep(h_clean_m,nlp,pos_to_remove=[""],normed=True,lemmatize=False)

    t_vectors=ut.get_matrix_rep(t_clean_m,nlp, pos_to_remove=[""],normed=False,lemmatize=False)
    h_vectors=ut.get_matrix_rep(h_clean_m,nlp, pos_to_remove=[""],normed=False,lemmatize=False)
    t_vectors_n=ut.get_matrix_rep(t_clean_m,nlp, pos_to_remove=[""],normed=True,lemmatize=False)
    h_vectors_n=ut.get_matrix_rep(h_clean_m,nlp, pos_to_remove=[""],normed=True,lemmatize=False)
    
    print(t_clean_m,h_clean_m)
    
    s1=t_clean_m.split()
    s2=h_clean_m.split()
    sinT=[]
    antT=[]
    HipT=[]
    sinH=[]
    antH=[]
    HipH=[]
    hipH=[]
    
    # encontrar bolsa de sinonimos de cada token
    for t in s1:
        sinT.append(bag_of_synonyms(t))
        # antT.append(bag_of_antonyms(t))
        # HipT.append(bag_of_hyperonyms(t))

    for h in s2:
        sinH.append(bag_of_synonyms(h))
        # antH.append(bag_of_antonyms(h))
        # HipH.append(bag_of_hyperonyms(h))
        # hipH.append(bag_of_hyponyms(h))
    # sobre los sinonimos encontrar sus hiperonimos,hiponimos y antonimos
    for k_t in range(len(sinT)):
        tempSet=set()
        tempSett=set()
        for s_ in sinT[k_t]:
            tempSet=tempSet.union(bag_of_antonyms(s_))
            tempSett=tempSett.union(bag_of_hyperonyms(s_))
        antT.append(tempSet)
        HipT.append(tempSett)
    for k_t in range(len(sinH)):
        tempSet=set()
        tempSett=set()
        tempSettt=set()
        for s_ in sinH[k_t]:
            tempSet=tempSet.union(bag_of_antonyms(s_))
            tempSett=tempSett.union(bag_of_hyperonyms(s_))
            tempSettt=tempSettt.union(bag_of_hyponyms(s_))
        antH.append(tempSet)
        HipH.append(tempSett)
        hipH.append(tempSettt)
    
    t_lem=ut.get_lemmas_(t_clean_m,nlp)
    h_lem=ut.get_lemmas_(h_clean_m,nlp)

    tp1,tp2,tp3=jaro_distance(t_lem, h_lem,sinT,sinH,HipT,hipH)
    new_data['Jaro-Winkler_rit'].append(tp1)
    #new_data['c_estructura'].append(tp2)
    new_data['nomatch'].append(tp3)
    
    #tp1,tp2=jaro_distance_relacionadas(t_lem, h_lem,sinT,HipT,sinH,hipH)
    #new_data['Jaro-Winkler_relacionadas'].append(tp1)
    #new_data['c2_estructura'].append(tp2)

    #tp1,tp2=jaro_distance_contra(t_lem, h_lem,antT,HipT,antH,HipH)
    #new_data['Jaro-Winkler_contra'].append(tp1)
    #new_data['c1_estructura'].append(tp2)
    
    # ngrams en 
    t_ngrams=get_grams_entities(textos[i])
    h_ngrams=get_grams_entities(hipotesis[i])

    print(t_ngrams,h_ngrams)
    new_data['bi_ent_trip_t'].append(t_ngrams)
    new_data['bi_ent_trip_h'].append(h_ngrams)

    lenh=len(set(h_ngrams))
    lent=len(set(t_ngrams))

    if lenh!=0 and lent!=0:
        new_data['bi_ent_trip_rel'].append(len(set(t_ngrams).intersection(set(h_ngrams)))/lenh)
    else:
        new_data['bi_ent_trip_rel'].append(0)

    #entidades
    doc = nlp(textos[i])
    entidadesT=[]
    for entity in doc.ents:
        #print(entity.text, entity.label_)
        entidadesT.append(entity.text)
    doc = nlp(hipotesis[i])

    print("Entidades T: ",entidadesT)

    entidadesH=[]
    for entity in doc.ents:
        #print(entity.text, entity.label_)
        entidadesH.append(entity.text)

    print("Entidades H: ",entidadesH)

    lenh=len(set(entidadesH))
    lent=len(set(entidadesT))

    if lenh!=0 and lent!=0:
        new_data['entidades'].append(len(set(entidadesT).intersection(set(entidadesH)))/lenh)
    else:
        new_data['entidades'].append(0)


    # Obtencion de matriz de alineamiento, matriz de move earth y mutual information
    ma=np.dot(t_vectors_n,h_vectors_n.T)
    #print(len(t_vectors_n),len(h_vectors_n),len(t_clean),len(h_clean))
    m_earth,m_mi=wasserstein_mutual_inf(t_vectors_n,h_vectors_n,t_clean,h_clean)
    ma=pd.DataFrame(ma,index=t_clean,columns=h_clean)

    # Calculamos la entropia inicial de la matriz de distancias coseno sobre tokens de T y H
    new_data['entropia_total'].append(entropia(ma.round(1).values.flatten())) 
    
    new_data['list_bigram'].append(bi_grams(t_clean,h_clean))
    #new_data['list_trigram'].append(tri_grams(t_clean,h_clean))
    #new_data['list_cuatrigram'].append(cuatri_grams(t_clean,h_clean))


    ###### BORRADO DE COSAS QUE NO OCUPO, SOLO NOS QUEDAMOS CON INFORMACIÓN DE TIPOS DE PALABRA: NOUN, VERB, ADJ Y ADV
    # TAMBIÉN OMITIMOS EL VERBO BE DEBIDO A QUE POR LO REGULAR SE UTILIZA COMO AUXILIAR Y ES UN VERBO COPULATIVO
    # sirve para construir la llamada predicación nominal del sujeto de una oración: 
    # #el sujeto se une con este verbo a un complemento obligatorio llamado atributo que por lo general determina 
    # alguna propiedad, estado o equivalencia del mismo, por ejemplo: "Este plato es bueno". "Juan está casado".
    c_compatibilidad=0
    c_incompatibilidad=0
    c_rel_concep=0
    b_col=[0]

    new_data['list_T'].append(ma.shape[0])
    new_data['list_M'].append(ma.shape[1])
    new_data['listas_malign'].append(ma)
    #print(ma.index,ma.columns)
    col=ma.columns
    borrar=[]
    indexes=ma.index
    for c in col:
        if "{null," in str(c) or "{be,VERB" in str(c) or (",NUM"  not in str(c) and "not," not in str(c) and "PRON" not in str(c) and "NOUN" not in str(c) and "VERB" not in str(c) and "ADJ" not in str(c) and "ADV" not in str(c)):
            borrar.append(c)     
            c_compatibilidad+=1   
        elif str(c) in indexes:
            borrar.append(c)
    #borrar_i=[]
    #for index in indexes:
    #    if "{null," in str(index) or "{be,VERB" in str(index) or (",NUM"  not in str(index) and "not," not in str(index) and "PRON" not in str(index) and "NOUN" not in str(index) and "VERB" not in str(index) and "ADJ" not in str(index) and "ADV" not in str(index)):
    #        borrar_i.append(index) 
    #    elif str(index) in col:
    #        borrar_i.append(index) 
    
    ma=ma.drop(borrar,axis=1)
    #ma=ma.drop(borrar_i,axis=0)
    m_earth=m_earth.drop(borrar,axis=1)
    #m_earth=m_earth.drop(borrar_i,axis=0)
    m_mi=m_mi.drop(borrar,axis=1)
    #m_mi=m_mi.drop(borrar_i,axis=0)
    
    b_col.extend(borrar)
    

    # ELIMINAMOS INFORMACIÓN DONDE SE CORRESPONDAN EN LEMMAS, TENGA UN PRODUCTO IGUAL A 1 Y SEAN IGUALES LOS INDICES Y COLUMNAS
    # SI EL VALOR ES IGUAL A 1 SIGNIFICA QUE ES LA MISMA PALABRA, O SON SINONIMOS
    borrar=[]
    borrar_i=[]
    col=ma.columns
    for index,strings in ma.iterrows():
        lema_i=str(index).split("{")[1].split(",")[0]
        for c in col:
            if index==c:
                borrar_i.append(index)
                borrar.append(c)
                c_compatibilidad+=1
            # if strings[c]>=1:
            #     borrar_i.append(index)
            #     borrar.append(c)
            lema_c=str(c).split("{")[1].split(",")[0]
            if lema_i == lema_c:
                borrar_i.append(index)
                borrar.append(c)
                c_compatibilidad+=1
    ma=ma.drop(borrar,axis=1)
    #ma=ma.drop(borrar_i,axis=0)
    m_earth=m_earth.drop(borrar,axis=1)
    #m_earth=m_earth.drop(borrar_i,axis=0)
    m_mi=m_mi.drop(borrar,axis=1)
    #m_mi=m_mi.drop(borrar_i,axis=0)
    b_col.extend(borrar)
    #primera vuelta ---------------------------------------------------------------------------------
    # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
    
    n_index = ma.shape[0]
    n_columns = ma.shape[1]
    pasada=0

    parejas=""

    print(ma.index,ma.columns)
    while n_columns>0 and pasada<3:
        borrar=[]
        a = ma.idxmax().values
        b = ma.columns
        for j in range(len(a)):
            print(a[j],b[j])
            wt_s=str(a[j]).split("{")[1].split(",")[0]
            wh_s=str(b[j]).split("{")[1].split(",")[0]
            sin_t=bag_of_synonyms(wt_s)
            ant_t=bag_of_antonyms(wt_s)
            hipe_t=bag_of_hyperonyms(wt_s)
            hipo_t=bag_of_hyponyms(wt_s)
            sin_h=bag_of_synonyms(wh_s)
            ant_h=bag_of_antonyms(wh_s)
            hipe_h=bag_of_hyperonyms(wh_s)
            hipo_h=bag_of_hyponyms(wh_s)
            #COMPATIBILIDAD SEMÁNTICA
            if len(sin_t.intersection(sin_h))>0 or wt_s in sin_h or wh_s in sin_t:
                #print(a[j],b[j],sin_t.intersection(sin_h),"sinonimos sinonimos")
                c_compatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " SS "+ b[j]+ " | "
            elif len(ant_t.intersection(ant_h))>0:
                #print(a[j],b[j],ant_t.intersection(ant_h),"antonimos de antonimos")
                c_compatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " AA "+ b[j]+ " | "
            elif len(hipe_t.intersection(sin_h))>0:
                #print(a[j],b[j],hipe_t.intersection(sin_h),"hiperonimo - sinonimos")
                c_compatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " HS "+ b[j]+ " | "
            elif len(hipo_t.intersection(hipo_h))>0:
                #print(a[j],b[j],hipo_t.intersection(hipo_h),"hiponimo - hiponimos")
                c_compatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " hh "+ b[j]+ " | "
            elif len(sin_t.intersection(hipo_h))>0:
                #print(a[j],b[j],sin_t.intersection(ant_t),"sinonimos hiponimos")
                c_compatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " sh "+ b[j]+ " | "
            elif wt_s in hipo_h:
                #print(a[j],b[j],wt_s,"generalidad")
                c_compatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " wh "+ b[j]+ " | "
            #INCOMPATIBILIDAD SEMÁNTICA
            if wt_s in hipe_h:
                #print(a[j],b[j],wt_s,"especifididad")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " wH "+ b[j]+ " | "
            elif len(sin_t.intersection(ant_h))>0:
                #print(a[j],b[j],sin_t.intersection(ant_t),"sinonimos antonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " SA "+ b[j]+ " | "
            elif len(sin_t.intersection(hipe_h))>0:
                #print(a[j],b[j],sin_t.intersection(ant_t),"sinonimos hiperonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " SH "+ b[j]+ " | "
            elif len(ant_t.intersection(sin_h))>0 or wh_s in ant_t:
                #print(a[j],b[j],ant_t.intersection(sin_h),wh_s,ant_t,"antonimos sinonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " AS "+ b[j]+ " | "
            elif len(ant_t.intersection(hipe_h))>0:
                #print(a[j],b[j],ant_t.intersection(hipe_h),"antonimos hiperonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " AH "+ b[j]+ " | "
            elif len(ant_t.intersection(hipo_h))>0:
                #print(a[j],b[j],ant_t.intersection(hipo_h),"antonimos hiponimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " Ah "+ b[j]+ " | "
            elif len(hipe_t.intersection(ant_h))>0:
                #print(a[j],b[j],hipe_t.intersection(ant_h),"hiperonimos antonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " HA "+ b[j]+ " | "
            elif len(hipe_t.intersection(hipe_h))>0:
                #print(a[j],b[j],hipe_t.intersection(hipe_h),"hiperonimos hiperonimos | cohiponimia")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " HH "+ b[j]+ " | "
            elif len(hipo_t.intersection(sin_h))>0:
                #print(a[j],b[j],hipo_t.intersection(sin_h),"hiponimos sinonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " hS "+ b[j]+ " | "
            elif len(hipo_t.intersection(ant_h))>0:
                #print(a[j],b[j],hipo_t.intersection(ant_h),"hiponimos antonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " hA "+ b[j]+ " | "
            elif len(hipo_t.intersection(hipe_h))>0:
                #print(a[j],b[j],hipo_t.intersection(hipe_h),"hiponimos hiperonimos")
                c_incompatibilidad+=1
                borrar.append(b[j])
                parejas = parejas + a[j] + " hH "+ b[j]+ " | "
            # match,grupo,relacion = compatibilidad_semantica(a[j],b[j])
            # if match and grupo==1:
            #     borrar.append(b[j])
            #     c_compatibilidad+=1
            #     parejas = parejas + a[j] + " - " + relacion + " - "+ b[j]+ " | "
            # elif match and grupo==2:
            #     borrar.append(b[j])
            #     c_incompatibilidad+=1
            #     parejas = parejas + a[j] + " - " + relacion + " - "+ b[j]+ " | "
            # elif match and grupo==3:
            #     borrar.append(b[j])
            #     c_rel_concep+=1
            #     parejas = parejas + a[j] + " - " + relacion + " - "+ b[j]+ " | "
        ma = ma.drop(borrar,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        pasada+=1
        b_col.extend(borrar)
    b_index=[0]
    #   ALMACENAMIENTO DE TODA LA INFORMACIÓN PROCESADA DE CARACTERÍSTICAS
    m_distancia = obtener_distancia(t_vectors,h_vectors,t_clean,h_clean,b_col,b_index)
    
    new_data['distancias'].append(m_distancia.max().sum()) #cambie de maximas a sumas
    #m_earth=m_earth*m_distancia
    if ma.shape[1]==0:
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
    new_data['list_rel_con'].append(c_rel_concep)
    new_data['list_relaciones'].append(parejas)
    new_data['listas_malignf'].append(ma)
    new_data['list_m'].append(ma.shape[1])

    st=""
    sh=""
    for t__1 in ma.index:
        st=st+" "+str(t__1).split("{")[0]
    for t__2 in ma.columns:
        sh=sh+" "+str(t__2).split("{")[0]
    doc1 = nlp(st)
    doc2 = nlp(sh)

    if sh!="" and st!="":
        new_data['similitud_faltantes'].append(doc1.similarity(doc2))
    elif st!="" and sh=="":
        new_data['similitud_faltantes'].append(1)
    else:
        new_data['similitud_faltantes'].append(0)
    new_data['clases'].append(prueba.at[i,"gold_label"])
fin = time.time()

#clases=prueba["gold_label"].values

#temp1 =[sumas,distancias,entropia_total,entropias,mutinf,mearts,max_info,similitud_faltantes,list_comp,list_incomp,list_rel_con,list_M,list_m,list_T,list_relaciones,listas_malign,listas_malignf,list_bigram,list_trigram,list_cuatrigram,clases]
#df_resultados = pd.DataFrame(temp1,columns=["suma","distancias","entropia_total","entropias","mutual_info","m_earth","max_info_p","sim_faltantes","Compatibilidad","Incompatibilidad","Rel_conceptuales","Shape Origin","Shape Finish","Total T","Match","Ma","Maf","bigram","trigram","cuatrigram","CLASS"])
df_resultados = pd.DataFrame(new_data)
#df_resultados.to_csv("salida/final/"+sys.argv[1]+".csv",index=False)
df_resultados.to_pickle("salida/pruebaCBS/"+sys.argv[1]+"_.pickle")

print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")