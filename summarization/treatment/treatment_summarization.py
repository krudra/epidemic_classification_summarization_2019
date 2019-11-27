# - *- coding: utf- 8 - *-
#!/usr/bin/python2

import sys
from collections import Counter
import re
from gurobipy import *
import gzip
import networkx as nx
from textblob import *
import time
import os
from happyfuntokenizing import *
import time
import codecs
import math
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic, genesis
import numpy as np
import aspell
from itertools import cycle
from operator import itemgetter


Tagger_Path = 'SET_YOUR_PATH/ark-tweet-nlp-0.3.2/' # set tagger path
ASPELL = aspell.Speller('lang', 'en')
WORD = re.compile(r'\w+')
cachedstopwords = stopwords.words("english")
METAMAP_PATH = '../DICTIONARY/Metamap-Abbreviation.txt'

def getdruginfo(DIC,METAMAP):
        dic = eval(DIC)
        A = []
	tag = ['phsu','clnd','drdd','antb','phsf']
        for k,v in dic.iteritems():
                test = 0
                flag = 0
                for x in k:
                        if ASPELL.check(x)==1:
                                test = 1
                                break
                if test==1:
                        for x in v:
				if x in tag:
					flag = 1
					break
                if flag==1:
                        for x in k:
                		A.append(x)
        return A

####################################################################
#  Inputs:
#   1. ifname: File containing details with Metamap outputs
#   2. orgname: Original Raw file Which contains Tweet id and Raw Tweet
#   3. ofname: Name of output file
#   4. SL: Summary Length constraint in terms of words
####################################################################

def treatment_summarization(ifname,orgname,ofname,SL):

	tok = Tokenizer(preserve_case=False)

        #########   Load Metamap Abbreviation ##################################
	METAMAP = {}
	fp = open(METAMAP_PATH,'r')
        for l in fp:
                wl = l.split('\t')
                w1 = wl[0].strip(' \t\n\r')
                w2 = wl[2].strip(' \t\n\r')
                METAMAP[w1] = w2
        fp.close()
        METAMAP['acty'] = 'Activity'


        ############# Load Raw tweets ##########################################
	TWEET = {}
	fp = open(orgname,'r')
	for l in fp:
		wl = l.split('\t')
		TWEET[wl[0].strip(' \t\n\r')] = wl[1].strip(' \t\n\r')
	fp.close()

	fp = open(ifname,'r')
	T = {}
	TW = {}
	index = 0
	count = 0
	word = {}
	t0 = time.time()
	KEYWORD = ['treat','treatment','treats','treating','treated']
	
	for l in fp:
		wl = l.split('\t')
		tid = wl[0].strip(' \t\n\r')
		text_org = TWEET[tid]
		text = wl[1].strip(' \t\n\r')
		tag = wl[4].strip(' \t\n\r')

		if tag=='treatment':
			count+=1
			DRG = getdruginfo(wl[3],METAMAP)
			unigram = tok.tokenize(text)
			KEY = []
			for x in unigram:
				if x in KEYWORD:
					try:
                                        	w = Word(x)
                                        	y = w.lemmatize("v")
                                	except Exception as e:
                                        	y = x
					KEY.append(y)
			temp = DRG + KEY
			################### Update word dictionary  ###################################
			for x in temp:
				if word.__contains__(x)==True:
					v = word[x]
					v+=1
					word[x] = v
				else:
					word[x] = 1
		
			################## Put tweet in the dictionary #################################

			T[index] = temp
			TW[index] = [tid,text_org,temp,int(wl[3])]
			index+=1
	fp.close()
		
	L = len(T.keys())
	weight = word
	tweet_cur_window = {}
        for i in range(0,L,1):
        	temp = TW[i]
                tweet_cur_window[temp[0].strip(' \t\n\r')] = [temp[1],temp[3],set(temp[2])]   ### Text, Length, Content words ###
	
	##################### Finally apply cowts ################################
        optimize(tweet_cur_window,weight,ofname,SL)
        print('Summarization done: ',ofname)
        t1 = time.time()
        print('Time Elapsed: ',t1-t0)

def optimize(tweet,weight,ofname,L):

	################################ Extract Tweets and Content Words ##############################
	word = {}
	tweet_word = {}
	tweet_index = 1
	for  k,v in tweet.iteritems():
		set_of_words = v[2]
		for x in set_of_words:
			if word.__contains__(x)==False:
				if weight.__contains__(x)==True:
					p1 = round(weight[x],4)
				else:
					p1 = 0.0
				word[x] = p1

		tweet_word[tweet_index] = [v[1],set_of_words,v[0]]  #Length of tweet, set of content words present in the tweet, tweet itself
		tweet_index+=1

	############################### Make a List of Tweets ###########################################
	sen = tweet_word.keys()
	sen.sort()
	entities = word.keys()
	print(len(sen),len(entities))

	################### Define the Model #############################################################

	m = Model("sol1")

	############ First Add tweet variables ############################################################

	sen_var = []
	for i in range(0,len(sen),1):
		sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))

	############ Add entities variables ################################################################

	con_var = []
	for i in range(0,len(entities),1):
		con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))

	########### Integrate Variables ####################################################################
	m.update()

	P = LinExpr() # Contains objective function
	C1 = LinExpr()  # Summary Length constraint
	C4 = LinExpr()  # Summary Length constraint
	C2 = [] # If a tweet is selected then the content words are also selected
	counter = -1
	for i in range(0,len(sen),1):
		C1 += tweet_word[i+1][0] * sen_var[i]
		v = tweet_word[i+1][1] # Entities present in tweet i+1
		C = LinExpr()
		flag = 0
		for j in range(0,len(entities),1):
			if entities[j] in v:
				flag+=1
				C += con_var[j]
		if flag>0:
			counter+=1
			m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))

	C3 = [] # If a content word is selected then at least one tweet is selected which contains this word
	for i in range(0,len(entities),1):
		P += word[entities[i]] * con_var[i]
		C = LinExpr()
		flag = 0
		for j in range(0,len(sen),1):
			v = tweet_word[j+1][1]
			if entities[i] in v:
				flag = 1
				C += sen_var[j]
		if flag==1:
			counter+=1
			m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))

	counter+=1
	m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))


	################ Set Objective Function #################################
	m.setObjective(P, GRB.MAXIMIZE)

	############### Set Constraints ##########################################

	fo = codecs.open(ofname,'w','utf-8')
	try:
		m.optimize()
		for v in m.getVars():
			if v.x==1:
				temp = v.varName.split('x')
				if len(temp)==2:
					fo.write(tweet_word[int(temp[1])][2])
					fo.write('\n')
	except GurobiError as e:
    		print(e)
		sys.exit(0)

	fo.close()

def main():
	try:
		_, ifname, orgname, ofname, SL = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	treatment_summarizatino(ifname,orgname,ofname,int(SL))

if __name__=='__main__':
	main()
