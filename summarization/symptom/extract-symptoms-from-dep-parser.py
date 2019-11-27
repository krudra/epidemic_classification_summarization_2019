import os
import codecs
import operator

## Inputs

# Folder of dataset files - each file contains symptom-classified tweets
basepath_1 = "Tweebo-Parser-Input/Symptoms"

# Output of [Tweebo Parser](https://github.com/SowaLabs/Tweebo) run on files in **basepath_1**
basepath = "Tweebo-Parser-Output/Symptoms"

# Symptom DICTIONARY Path
dictionary_path = "../DICTIONARY/symptom-dictionary.txt"

# Subjectivity DICTIONARY for negative words - https://github.com/kuitang/Markovian-Sentiment/blob/master/data/subjclueslen1-HLTEMNLP05.tff
sub_path = "../DICTIONARY/subjclueslen1-HLTEMNLP05.tff"


# Other negative words (manually created)
negex_path = "../DICTIONARY/negative-words.txt"


def main():

	dictionary = set()

	file = codecs.open(dictionary_path, 'r', 'utf-8')

	for row in file:
		dictionary.add(row.strip().lower())


	negative_dict = set()

	file = codecs.open(sub_path, 'r', 'utf-8')

	for row in file:
		wl = row.split()
		Type = wl[0].split('=')[1].strip(' \t\n\r')
		word = wl[2].split('=')[1].strip(' \t\n\r')
		Tag = wl[5].split('=')[1].strip(' \t\n\r')
		if Type=='strongsubj':
			
			if Tag=='negative':
				negative_dict.add(word.lower())


	negative_phrase = set()

	file = codecs.open(negex_path,'r', 'utf-8')

	for row in file:
		negative_dict.add(row.strip().lower())


	files = os.listdir(basepath)

	for filename in files:

		symptoms = dict()

		negative_symp = set()

		amb_symp = set()

		dict_tweet = {}

		sym_tweet = set()
		file = codecs.open(basepath+'/'+filename, 'r', 'utf-8')

		tweet_file = codecs.open(basepath_1+'/'+filename, 'r', 'utf-8')

		tweet_list = tweet_file.read().split('\n')

		cnt = 0

		for row in file:
			row  = row.strip()
			if row == "": #line break; indicates end of previous tweet
				whole_tweet = tweet_list[cnt]
				# print(whole_tweet)
				cnt+=1

				for word_idx in sym_tweet:

					one_hop = set()
					sym_word = dict_tweet[word_idx][0].lower()
					f = 0
					for key in dict_tweet:
						idx = key
						word = dict_tweet[idx][0]
						postag = dict_tweet[idx][1]
						dep = dict_tweet[idx][2]

						if dep == word_idx:
							if word in negative_dict and word not in dictionary:
								if sym_word == "depression":
									print(sym_word, word, whole_tweet)
								f = 1
								if sym_word not in symptoms.keys():
									negative_symp.add(sym_word)
								else:
									symptoms.pop(sym_word, None)
									amb_symp.add(sym_word)
									if sym_word in negative_symp:
										negative_symp.remove(sym_word)
							else:
								one_hop.add(idx)

					for one_hop_idx in one_hop:
						for key in dict_tweet:
							idx = key
							word = dict_tweet[idx][0]
							postag = dict_tweet[idx][1]
							dep = dict_tweet[idx][2]

							if dep == one_hop_idx:
								if word in negative_dict and word not in dictionary:
									if sym_word == "depression":
										print(sym_word, word, whole_tweet)
									# print(word)
									f = 1
									if sym_word not in symptoms:
										negative_symp.add(sym_word)
									else:
										symptoms.pop(sym_word, None)
										amb_symp.add(sym_word)
										if sym_word in negative_symp:
											negative_symp.remove(sym_word)


					if f==0:
						if sym_word not in negative_symp:
							if sym_word in symptoms.keys():
								symptoms[sym_word] += 1
							else:
								symptoms[sym_word] = 1
						else:
							negative_symp.remove(sym_word)
							amb_symp.add(sym_word)
							symptoms.pop(sym_word, None)



				sym_tweet = set()


				dict_tweet = {}

			else: ### Tweet is continuing
				s = row.split('\t')
				
				idx = int(s[0])
				word = s[1].lower()
				postag = s[3]

				if word in dictionary:
					sym_tweet.add(idx)

				if s[6] == "_":
					dep = int(s[7])
				else:
					dep = int(s[6])

				dict_tweet[idx] = (word,postag, dep)
		


		print(filename)

		sorted_x = sorted(symptoms.items(), key=operator.itemgetter(1), reverse=True)

                print('Ranked list of symptoms')
		for elem in sorted_x:
			print(str(elem[0])+'\t'+str(elem[1]))

                print('Negative symptoms: {}'.format(negative_symp))
                print('Ambiguous symptoms: {}'.format(amb_symp))


if __name__ == "__main__":main()
