# - *- coding: utf- 8 - *-
#!/usr/bin/python2

import sys
import os
import random
import json
import re
import time
import codecs
import string
import networkx as nx
import math
from gurobipy import *
from operator import itemgetter
from happyfuntokenizing import *
from nltk.corpus import stopwords
from textblob import *
from nltk.stem.wordnet import WordNetLemmatizer

import numpy as np
import gzip
import pickle

cachedstopwords = stopwords.words("english")	# English Stop Words
Tagger_Path = 'SET_YOUR_PATH/ark-tweet-nlp-0.3.2/'	# Set Twitter Tagger Path
lmtzr = WordNetLemmatizer()		# Lemmatizer
METAMAP_PATH = '../DICTIONARY/Metamap-Abbreviation.txt'

####################################################################
#  Inputs:
#   1. ifname: File containing details with Metamap outputs
#   2. orgname: Original file Which contains Tweet id and Raw Tweet
#   3. placefile: Place file containing locations
#   4. ofname: Name of output file
#   5. SL: Summary Length constraint in terms of words
####################################################################

def death_report_summarization(ifname,orgname,placefile,ofname,SL):
	
	#########################################################################

	tok = Tokenizer(preserve_case=False)
	TAGREJECT = ['U','@','#','~','E','~',',']

        ################### Read Locations ######################################
	PLACE = {}
	fp = open(placefile,'r')
	for l in fp:
		PLACE[l.strip(' \t\n\r').lower()] = 1
	fp.close()

        ################### Read Original File ###################################
	TWEET = {}
	fp = open(orgname,'r')
	for l in fp:
		wl = l.split('\t')
		TWEET[wl[0].strip(' \t\n\r')] = wl[1].strip(' \t\n\r')
	fp.close()

        
        ###################### START PROCESSING MAIN FILE #########################
	Tweet = []
	fp = open(ifname,'r')
	fo = open('temp.txt','w')
	for l in fp:
		wl = l.split('\t')
		Tweet.append(wl)
		fo.write(TWEET[wl[0].strip(' \t\n\r')])
		fo.write('\n')
	fp.close()
	fo.close()

	command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
	os.system(command)

	t0 = time.time()	
	KEYWORD = ['death','dead','die','kill','demise','expire','dies','killed','expired']
	fp = open('tag.txt','r')
	temp = set([])
	L = 0
	index = 0
	count = 0
	Tword = {}
	T = {}
	TW = {}
	for l in fp:
		wl = l.split('\t')
		if len(wl)>1:
			word = wl[0].strip(' \t\n\r').lower()
			tag = wl[1].strip(' \t\n\r')
			if tag not in TAGREJECT:
				L+=1
			if tag=='$':
				temp.add(word)
			elif tag=='N' or tag=='^':
				try:
					w = lmtzr.lemmatize(word)
					word = w.lower()
					if word in KEYWORD:
						temp.add(word)
				except Exception as e:
					if word in KEYWORD:
						temp.add(word)
			elif tag=='V':
				try:
					w = Word(word.lower())
					x = w.lemmatize("v")
				except Exception as e:
					x = word.lower()
				word = x.lower()
				if word in KEYWORD:
					temp.add(word)
			elif PLACE.__contains__(word)==True:
				temp.add(word)
			else:
				pass
		else:
			for x in temp:
				if Tword.__contains__(x)==True:
					v = Tword[x]
					v+=1
					Tword[x] = v
				else:
					Tword[x] = 1
			XL = Tweet[count]
                        T[index] = temp
                        TW[index] = [XL[0].strip(' \t\n\r'),TWEET[XL[0].strip(' \t\n\r')],temp,int(XL[3])]
                        index+=1
			count+=1
			L = 0
			temp = set([])
	fp.close()
        
	L = len(TW.keys())
	tweet_cur_window = {}
        for i in range(0,L,1):
                temp = TW[i]
                tweet_cur_window[str(temp[0])] = [temp[1],temp[3],set(temp[2])]   ### Text, Length, Content words ###

	##################### Finally apply cowts ################################
        optimize(tweet_cur_window,Tword,ofname,SL)
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
		#P += sen_var[i]
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
		_, ifname, orgname, placefile, ofname, SL = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)

	death_report_summarization(ifname,orgname,placefile,ofname,int(SL))

if __name__=='__main__':
	main()
