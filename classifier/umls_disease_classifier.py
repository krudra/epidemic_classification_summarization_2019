# - *- coding: utf- 8 - *-
#!/usr/bin/python2

import sys
import os
import random
import json
import re
import numpy as np
import codecs
import string
from happyfuntokenizing import *
from nltk.corpus import stopwords
from textblob import *
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate

from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import gzip
import pickle


##################### DICTIONARY FILE PATH ####################

METAMAP_PATH = '../DICTIONARY/Metamap-Abbreviation.txt'
TRANSMISSION_PATH = '../DICTIONARY/transmission_term.txt'
TREATMENT_PATH = '../DICTIONARY/treatment_term.txt'
DEATH_PATH = '../DICTIONARY/death_term.txt'
PREVENT_PATH = '../DICTIONARY/prevent_term.txt'

###################### END OF DICTIONARY FILE PATH #############

cachedstopwords = stopwords.words("english")    # English Stop Words
Tagger_Path = 'SET_YOUR_PATH/ark-tweet-nlp-0.3.2/'    # Set your tagger path
lmtzr = WordNetLemmatizer()        # Lemmatizer

FULLTAG = ['N','^','S','Z','V','A','R','$','L','M','Y']


METAMAP = {}
TRANSMISSION = {}
TREATMENT = {}
DEATH = {}
PREVENT = {}
SYMPTOM = {}

##################################
# Reads Dictionary files and stores them in respective dictionary
##################################

def Read_Files():

    fp = open(METAMAP_PATH,'r')
    for l in fp:
        wl = l.split('\t')
        w1 = wl[0].strip(' \t\n\r')
        w2 = wl[2].strip(' \t\n\r')
        METAMAP[w1] = w2
    fp.close()
    METAMAP['acty'] = 'Activity'

    fp = open(TRANSMISSION_PATH,'r')
    for l in fp:
        TRANSMISSION[l.strip(' \t\n\r').lower()] = 1
    fp.close()
    
    fp = open(TREATMENT_PATH,'r')
    for l in fp:
        TREATMENT[l.strip(' \t\n\r').lower()] = 1
    fp.close()
    
    fp = open(DEATH_PATH,'r')
    for l in fp:
        DEATH[l.strip(' \t\n\r').lower()] = 1
    fp.close()
    
    fp = open(PREVENT_PATH,'r')
    for l in fp:
        PREVENT[l.strip(' \t\n\r').lower()] = 1
    fp.close()
    
def getprocedure(DIC,XT):
    dic = eval(DIC)
    tag = ['topp']
    flag = 0
    for k,v in dic.iteritems():
        test = 0
        for x in k:
            if x not in cachedstopwords:
                if XT.__contains__(x)==True:
                    if XT[x] in FULLTAG:
                        test = 1
                        break
                else:
                    test = 1
                    break
        if test==1:
            for x in v:
                if x in tag:
                    flag = 1
                    break
    return flag

def getsymptom(DIC,XT):
    dic = eval(DIC)
    flag = 0
    tag = ['phsf','sosy']
    for k,v in dic.iteritems():
        test = 0
        for x in k:
            if x not in cachedstopwords:
                if XT.__contains__(x)==True:
                    if XT[x] in FULLTAG:
                        test = 1
                        break
                else:
                    test = 1
                    break
        if test==1:
            for x in v:
                if METAMAP[x] == 'Symptom':
                    flag=1
                    break
    return flag

def getactivity(DIC,XT):
    dic = eval(DIC)
    flag = 0
    for k,v in dic.iteritems():
        test = 0
        for x in k:
            if x not in cachedstopwords:
                if XT.__contains__(x)==True:
                    if XT[x] in FULLTAG:
                        test = 1
                        break
                else:
                    test = 1
                    break
        if test==1:
            for x in v:
                if METAMAP[x] == 'Activity':
                    flag=1
                    break
    return flag

def getanatomy(DIC,XT):
    dic = eval(DIC)
    flag = 0
    tag = ['bdsu','blor','bpoc']
    for k,v in dic.iteritems():
        test = 0
        for x in k:
            if x not in cachedstopwords:
                if XT.__contains__(x)==True:
                    if XT[x] in FULLTAG:
                        test = 1
                        break
                else:
                    test = 1
                    break
        if test==1:
            for x in v:
                if x in tag:
                    flag = 1
                    break
    return flag

def getprevent(text):
    for x in text:
        if PREVENT.__contains__(x)==True:
            return 1
    return 0

def gettransmission(text):
    for x in text:
        if TRANSMISSION.__contains__(x)==True:
            return 1
    return 0

def gettreatment(text):
    for x in text:
        if TREATMENT.__contains__(x)==True:
            return 1
    return 0

def getdeath(text):
    for x in text:
        if DEATH.__contains__(x)==True:
            return 1
    return 0

####################################################################
#  #  Inputs:
#   1. ifname: File containing details with Metamap outputs and which need to be classified
#   2. modelname: Name of the trained model
#   3. ofname: Name of output file

####################################################################

def predict(ifname,modelname,ofname):
    
    ######################## LOAD DICTIONARIES ##############################

    Read_Files()

    #########################################################################

    tok = Tokenizer(preserve_case=False)

    CL = {}
    RCL = {}
    count = 0
    fp = open('out_class.txt','r')
    for l in fp:
        CL[l.strip(' \t\n\r')] = count
        count+=1
    fp.close()

    for k,v in CL.iteritems():
        RCL[v] = k
    
    tagreject = ['U','@','#','~','E','~',',']

    ''' Read Dataset '''
    fp = open(ifname,'r')
    fo = open('temp.txt','w')
    tweet = []
    for l in fp:
        wl = l.split('\t')
        fo.write(wl[1].strip(' \t\n\r') + '\n')
        tweet.append(wl)
    fp.close()
    fo.close()

    command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
    os.system(command)

    fp = open('tag.txt','r')
    s = ''
    ques = 0
    excl = 0
    count = 0
    dic = {}
    N = 0
    feature = []
    for l in fp:
        wl = l.split('\t')
        if len(wl)>1:
            word = wl[0].strip(' #\t\n\r').lower()
            tag = wl[1].strip(' \t\n\r')
            if tag not in tagreject:
                dic[word] = tag
                if tag=='N':
                    try:
                        w = lmtzr.lemmatize(word)
                        word = w
                    except Exception as e:
                        pass
                elif tag=='V':
                    try:
                        w = Word(word)
                        x = w.lemmatize("v")
                    except Exception as e:
                        x = word
                    word = x.lower()
                elif tag=='$':
                    N+=1
                else:
                    pass
                try:
                    s = s + word + ' '
                except Exception as e:
                    pass
            else:
                if tag==',':
                    if word.startswith('?')==True:
                        ques = 1
                    elif word.startswith('!')==True:
                        excl = 1
                    else:
                        pass
        else:
            unigram = list(tok.tokenize(s.strip(' ')))
            bigram = []
            if len(unigram)>=2:
                for i in range(0,len(unigram)-1,1):
                    s = unigram[i] + ' ' + unigram[i+1]
                    bigram.append(s)

            trigram = []
            if len(unigram)>=3:
                for i in range(0,len(unigram)-2,1):
                    s = unigram[i] + ' ' + unigram[i+1] + ' ' + unigram[i+2]
                    trigram.append(s)
            Ngram = unigram + bigram + trigram
            PROC = getprocedure(tweet[count][2],dic)
            SYMP = getsymptom(tweet[count][2],dic)
            TRAN = gettransmission(unigram)
            TREA = gettreatment(unigram)
            DEAT = getdeath(unigram)
            PRVT = getprevent(unigram)
            ACTV = getactivity(tweet[count][2],dic)
            ANAT = getanatomy(tweet[count][2],dic)
            NUM = 0
            if N>0:
                NUM = 1
            t = (PROC,PRVT,SYMP,ANAT,TRAN,TREA,DEAT)
            feature.append(t)
            s = ''
            ques = 0
            excl = 0
            N = 0
            count+=1
            dic = {}
    fp.close()

    train_model = joblib.load(modelname)
    predicted_label = train_model.predict(feature)
    predicted_proba = train_model.predict_proba(feature)
    fp = codecs.open(ifname,'r','utf-8')
    fo = codecs.open(ofname,'w','utf-8')
    count = 0
    for l in fp:
        s = l.strip(' \t\n\r') + '\t' + RCL[predicted_label[count]] + '\t' + str(max(predicted_proba[count]))
        fo.write(s+'\n')
        count+=1
    fp.close()
    fo.close()

def main():
    try:
        _, ifname, modelname, ofname = sys.argv
    except Exception as e:
        print(e)
        sys.exit(0)

    predict(ifname, modelname, ofname)
    print('Classification Done')

if __name__=='__main__':
    main()
