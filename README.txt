Paper Description

This paper proposes a method to classify and summarize tweets posted during an epidemic.

Paper link: https://link.springer.com/article/10.1007/s10796-018-9844-9

UMLS concept based classification and summarization of epidemic-related tweets

A. First tweets are classified into following classes --- 1. Symptom, 2. Prevention, 3. Transmission, 4. Treatment, 5. Death reports, 6. Not relevant

B. Symptom and Transmission classes contain information about both kinds of entities (related, not related). We explore the syntax of tweets to get both kinds of entities. Side by side, it also highlights ambiguous entities.

C. For Death reports, Prevention, and Treatment, we propose an ILP algorithm to summarize the information based on UMLS concepts.

Requirements

1. GUROBI ILP Solver [https://www.gurobi.com/]
2. Setup of QuickUMLS tool. [https://github.com/Georgetown-IR-Lab/QuickUMLS]
3. Twitter POS-Tagger and Parser [http://www.cs.cmu.edu/~ark/TweetNLP/]
4. Install ASPELL [https://github.com/WojciechMula/aspell-python]
5. Install nltk. Codes are tested on Python2.7

Folder strucutre and descriptions

1. classifier --- Contains the code for tweet classification
2. summarization --- contains the summarization code for different classes
3. DICTIONARY --- contains all the lexical resources

Data

Link for Ebola and MERS tweet-ids are present in the paper. 

Running Setup:

1. First apply QuickUMLS tool to extract concepts from tweets and prepare output file in the following format: tweet-id \t tweet-text \t metamap concept dictionary \t Length of the tweet

2. Apply classifier to the output file of step 1.

3. Finally, summarize tweets from different information classes. Separate tweets of different classes and apply corresponding summarization approach. Summarization folder contains five different sub-folders, one
 for each category.
