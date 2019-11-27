# - *- coding: utf- 8 - *-
#!/usr/bin/python2


import sys
import os
import random
import json
import re
import codecs
import string
import networkx as nx
from operator import itemgetter
from happyfuntokenizing import *
from nltk.corpus import stopwords
from textblob import *
from nltk.stem.wordnet import WordNetLemmatizer
import aspell
import numpy as np
import gzip
import pickle

ASPELL = aspell.Speller('lang', 'en')

cachedstopwords = stopwords.words("english")	# English Stop Words
Tagger_Path = 'SET_YOUR_PATH/ark-tweet-nlp-0.3.2/'	# Twitter pos tagger path
lmtzr = WordNetLemmatizer()		# Lemmatizer

def negatedContextCount(s):
        negation = re.compile("(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't")
        clauseLevelPunctuation = re.compile("^[.:;!?]$")
        tok = Tokenizer(preserve_case=False)
        tokenized = tok.tokenize(s)
        count= 0
        for token in tokenized :
                if negation.search(token) :
                        for t in tokenized[tokenized.index(token) :] :
                                if clauseLevelPunctuation.search(t) :
                                        break
                                count+=1
                        break
        if count>=1:
                return 1
        return count

####################################################################
#  Inputs:
#   1. ifname_parsed: Parsed tweets from transmission category (Tweebo parser)
#   2. positive_ofname: Output file for medium responsible for transmission
#   3. negative_ofname: Output file for medium not responsible for transmission
####################################################################

def transmission_summarization(ifname_parsed,positive_ofname,negative_ofname):
	
	TAGREJECT = ['U','@','#','~','E','~',',']

        fp = open(ifname_parsed,'r')
	medium = {}
	medium_neg = {}
	pos_tweet = set([])
	neg_tweet = set([])
	dic = {}
	
	for l in fp:
		wl = l.split('\t')
                if len(wl)==8:
                        seq = int(wl[0])
                        word = wl[1].strip(' #\t\n\r').lower()
                        tag = wl[4].strip(' \t\n\r')
                        dep = wl[6].strip(' \t\n\r')
                        if dep=='_':
                                dep = int(wl[7].strip(' \t\n\r'))
                        else:
                                dep = int(wl[6])

			if tag=='N':
				try:
					w = lmtzr.lemmatize(word)
					word = w.lower()
				except Exception as e:
					pass
			elif tag=='V':
				try:
					w = Word(word.lower())
					x = w.lemmatize("v")
				except Exception as e:
					x = word.lower()
				word = x.lower()
			else:
				pass
                        temp = [word,tag,dep]
                        dic[seq] = temp
                else:
                        temp = dic.keys()
                        temp.sort()
                        G = nx.Graph()
                        for x in temp:
                                G.add_node(x)
                        for x in temp:
                                dep = dic[x][2]
                                if dep!=-1 and dep!=0 and dic[x][1] not in TAGREJECT:
                                        G.add_edge(dep,x)
                        temp = sorted(nx.connected_components(G), key = len, reverse=True)
                        
                        for i in range(0,len(temp),1):
                                comp = temp[i]
				flag = 0
				TR = []
				for x in comp:
					if dic[x][0]=='transmission' or dic[x][0]=='transmit':
						TR.append(x)
				NEGCON = 0
				
				if len(TR)>0:
					s = ''
					for x in comp:
						s = s + dic[x][0] + ' '
					s = s.strip(' ')
					NEGCON = negatedContextCount(s)

				for x in comp:
                                	if dic[x][1]=='N' and ASPELL.check(dic[x][0])==1 and len(dic[x][0])>2:
                                        	shp = []
                                                for y in TR:
                                                	if x!=y:
                                                        	shp.append(nx.shortest_path_length(G,source=x,target=y))
						shp.sort()
                                                if len(shp)>0:
                                                	if shp[0]<=2:
								if NEGCON==0:
									pos_tweet.add(s)
									if medium.__contains__(dic[x][0])==True:
										v = medium[dic[x][0]]
										v+=1
										medium[dic[x][0]] = v
									else:
										medium[dic[x][0]] = 1
								else:
									neg_tweet.add(s)
									if medium_neg.__contains__(dic[x][0])==True:
										v = medium_neg[dic[x][0]]
										v+=1
										medium_neg[dic[x][0]] = v
									else:
										medium_neg[dic[x][0]] = 1
			dic = {}

	fp.close()


        ###### Rank medium ######
	temp = []
	for k,v in medium.iteritems():
		temp.append((k,v))
	temp.sort(key=itemgetter(1),reverse=True)
	
        fo = open(positive_ofname,'w')
	for x in temp:
		fo.write(x + '\n')
	fo.close()
	

        ###### Rank negative medium ######
	temp_neg = []
	for k,v in medium_neg.iteritems():
		temp_neg.append((k,v))
	temp_neg.sort(key=itemgetter(1),reverse=True)

	fo = open(negative_ofname,'w')
	for x in temp_neg:
		s = x[0] + '\t' + str(x[1])
		fo.write(s+'\n')
	fo.close()

def main():
	try:
		_, ifname_parsed, positive_ofname, negative_ofname = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)

	transmission_summarization(ifname_parsed, positive_ofname, negative_ofname)

if __name__=='__main__':
	main()
