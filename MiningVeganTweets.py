import glob
import nltk
import json,pprint,re,csv,string,nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pprint import pprint
import itertools
import matplotlib.pyplot as plt
import collections
import numpy as np
from wordcloud import WordCloud
from nltk.metrics.distance import jaccard_distance
from cluster import HierarchicalClustering
import gensim,logging
from gensim.models import word2vec
from PyDictionary import PyDictionary
from nltk.corpus import words, brown
from collections import Counter
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora,models
import pprint
from PIL import Image
#Data retrieved from Twitter
def cleaning(tweets):

	additional_stopwords=['amp','new','#vegan', '@','rt','vegan','#','veganhour','govegan','plantbased','recipes','please','veganism','food','vegetarian','recipe','chocolate','vegano','veganlife','vegandiet','raw','day','vegangirl','glutenfree','today','via','veggie','vegans','follow','dinner','make','diet','im']
	stop=set(stopwords.words('english'))
	#target=open('tweets.txt','w')
	#Remove repeated tweets
	tweets=list(set(tweets))
	sentences=[]
	#print count,len(tweets)
	for i in range(len(tweets)):
		# All tweets in lower case
		tweets[i]=tweets[i].lower() 
		#Removing unicodes
		tweets[i] = tweets[i].encode('ascii','ignore')
		# Removing URL 
		tweets[i] = re.sub(r'http\S+', '', tweets[i])
		#New Line characters
		tweets[i] =re.sub('\n','', tweets[i])
		#Removing rt @ till the next whitespace
		tweets[i] =re.sub('rt\s@\S+','',tweets[i])
		#Strip spaces
		tweets[i]=tweets[i].strip()
		#Remove extra spaces
		tweets[i]=re.sub(' +',' ',tweets[i])
		#Remove punctuations
		#Removes everything including '#'tweets[i]=tweets[i].translate(None, string.punctuation)
		tweets[i]=re.sub(r'[/!$%^&*():,-;=?~]', ' ', tweets[i])
		#Removes ""
		tweets[i]=re.sub(r'"', '', tweets[i])
		#Removes '
		tweets[i]=re.sub(r'\'', '', tweets[i])
		#Remove #vegan
		#tweets[i] =re.sub('#vegan','', tweets[i])
		#tweets[i] =re.sub('@','', tweets[i])
		#tweets[i] =re.sub('amp', '')
		#Removing stopwords
		tweets[i]=' '.join([j for j in word_tokenize(tweets[i]) if j not in stop])
		tweets[i]=' '.join([j for j in word_tokenize(tweets[i]) if j not in additional_stopwords])
		#Removes space after #s
		tweets[i]=re.sub(r'#\s','#',tweets[i])
		sentences.append(tweets[i].split("."))
	#print sentences[:10]
	return tweets,sentences
#sample=tweets[0]
def lda1(words):
	#en_stop=get_stop_words('en')

	p_stemmer=PorterStemmer()
	stemmed_tokens=[p_stemmer.stem(str(i)) for i in words]
	# texts=[]
	# foiiiiir tweet in tweets:
	# 	texts.append(tweet.split())
		
	
	#print texts[:100]
	#for i in range(len(words)):
	#	words[i]=str[words[i]]
	#print type(words)
	dictionary=corpora.Dictionary([stemmed_tokens])
	#print dictionary[:100]
	corpus=[dictionary.doc2bow(stemmed_tokens)]
	#print corpus[0]
	ldamodel=gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary,passes=20)
	return ldamodel.print_topics(num_topics=20,num_words=5) 

def Dict1(tweetsVegan):
	#dictionary=PyDictionary()

	model=gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
	words = ' '.join(tweetsVegan).split()
	
	hashTable=[]
	try:
		length=len(words)
		#print 'intial length'
		#print length
		for i in range(length):
			#print i,length
			#print word
			#if word not in hashTable:
			#oldLength=len(words)
			if words[i] not in hashTable:
				similarWords=model.most_similar(words[i])
				if similarWords:
					hashTable.append(words[i])
					#print word,similarWords
					similarWords.append(words[i])
					for similarWord in similarWords:
						if similarWord in words:
							for j in range(len(words)):
								if similarWord==words[j]:
									words[j]=words[i]
							
				else:
				 	words.remove(words[i])
				 	i=i-1
				 	length=length-1

		# file=open("words.txt","w")
		# file.write(' '.join(words))
		# file.close()
				# 	print word
				# 	#possibility=
				# 	if split_hashtag_to_words_all_possibilities(word):
				# 		for word in split_hashtag_to_words_all_possibilities(word)[0]:
				# 			print word
				# 			words.append(word)
					
				
				
				#similarWords=[word]
			#similarWords.append(tuple((word,0)))
			#synonyms=[word]
			
			#print similarWords

			
			#newLength=len(words)
			#print words
			
			#print hashTable[word]
	except KeyError,IndexError:
		print 'Not in vocab'

	#print ""

	return Counter(words),words

def Dict2(tweetsVegan):
	try:
		model=gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
		words = ' '.join(tweetsVegan).split()
		listOfWords=[]
		for i in range(len(words)):
			


			if words[i] not in listOfWords:
				listOfWords.append(words[i])
				for j in range(len(words)):
					if model.similarity(words[i],words[j]) > 0.85:
						words[j]==words[i]

	except KeyError:
		print "Word not found"
	for i in range(len(words)):
		if words[i]=='healthyfood' or  words[i]=='health' or words[i]=='nutrition' or words[i]=='wellness' or words[i]=='ill':
			words[i]='healthy' 
		if words[i]=='animalrights' or words[i]=='animal' or words[i]=='animals' or words[i]=='veganfortheanimals' or words[i]=='veganfortheanimal'  or words[i]=='meatlessmonday' or words[i]=='crueltyfree' or words[i]=='banhorseslaughter' or words[i]=='dairyfree' or words[i]=='mercyforanimals' or words[i]=='slaughter' or words[i]=='cruelty' or words[i]=='rescue' or words[i]=='killing' or words[i]=='abuse' or words[i]=='compassion' or words[i]=='kind':
			words[i]='animalcruelty'
		if words[i]=='yoga' or words[i]=='bodybuilding' or words[i]=='energy':
			words[i]='fitness'
		if words[i]=='skins' or words[i]=='acne' or words[i]=='skin':
			words[i]='skincare'
		if words[i]=='fat' or words[i]=='excercise' or words[i]=='body' or words[i]=='weight' or words[i]=='obesity' or words[i]=='loseweight' or words[i]=='weightlos':
			words[i]='weightloss'
		if words[i]=='green'or words[i]=='organic' or words[i]=='planet':
			words[i]='environment'
	return Counter(words),words

def Word2Vector(sentences,tweetsVegan):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

	s=[]
	for sentence in sentences:
		#print sentence
		s.append(sentence[0].split())
	model = gensim.models.Word2Vec(s,min_count=1, size=500)
	words = ' '.join(tweetsVegan).split()
	hashTable={}
	try:
		while words:
			word=words[0]
			print word
			#if word not in hashTable:
			oldLength=len(words)
			similarWords=model.most_similar(word)
			#similarWords.append(tuple((word,0)))
			synonyms=[word]
			if similarWords:
				for similarWord in similarWords:
					synonyms.append(similarWord[0])
			#print "check"
			print similarWords

			for similarWord in synonyms:
				if similarWord in words:
					words=list(filter(lambda a: a!=similarWord,words))
			newLength=len(words)
			#print words
			hashTable[word]= oldLength-newLength
			#print hashTable[word]
	except KeyError:
		print 'Not in vocab'

	#print ""
	return hashTable

def split_hashtag_to_words_all_possibilities(hashtag):
	word_dictionary = list(set(words.words()))

	for alphabet in "bcdefghjklmnopqrstuvwxyz":
		word_dictionary.remove(alphabet)


	all_possibilities = []
	
	split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag)+1))]
	possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]
	
	for split_pos in possible_split_positions:
		split_words = []
		word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]
		
		if word_2 in word_dictionary:
			split_words.append(word_1)
			split_words.append(word_2)
			all_possibilities.append(split_words)

			another_round = split_hashtag_to_words_all_possibilities(word_2)
				
			if len(another_round) > 0:
				all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
		else:
			another_round = split_hashtag_to_words_all_possibilities(word_2)
			
			if len(another_round) > 0:
				all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
	
	return all_possibilities

def generate_unigrams(tweets):
	unigrams=[]
	for i in range(len(tweets)):
		words=tweets[i].split()
		unigrams.append(words)
	unigrams=list(itertools.chain(*unigrams))

def generate_bigrams(words):
	text=' '.join(words)
	print text
	bigrams=[]
	#for i in range(len(tweets)):
	pairs = [b for b in zip(text.split(" ")[:-1], text.split(" ")[1:])]
	for pair in pairs:
		bigram=' '.join(pair)
		bigrams.append(bigram)		
	#bigrams=list(itertools.chain(*bigrams))
	return bigrams

def wc(max_words):
	wordcloud = WordCloud(width = 1000, background_color="white",height = 500,margin=2,max_words=500,mask=np.array(Image.open("Vegan/Leaf.png")))
	#print ' '.join(words)
	# with open("words.txt","r") as f:
	# 	text=f.read()

	wordcloud.generate(' '.join(words))
	wordcloud.to_file('vegan.png')
	
	# plt.figure(figsize=(15,8))
	# plt.imshow(wordcloud)
	# plt.axis("off")
	# plt.show()

# Define a scoring function
def score(unigrams1, unigrams2):
    return DISTANCE(set(list(unigrams1)), set(list(unigrams2)))

DISTANCE = jaccard_distance

def cluster(unigrams):
	DISTANCE_THRESHOLD = 0.2
    # Feed the class your data and the scoring function
	hc = HierarchicalClustering(unigrams,score)
    # Cluster the data according to a distance threshold
	clusters = hc.getlevel(DISTANCE_THRESHOLD)
    # Remove singleton clusters
	clusters = [c for c in clusters if len(c) > 20]
	return clusters

VeganData=[]
import os, json
import pandas as pd
tweetCount=[]
fileCount=0
path_to_json = 'Vegan/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
for js in json_files:
	fileCount=fileCount+1
	with open(os.path.join(path_to_json, js)) as json_file:
        #o something with your json; I'll just print
		vegan=[]
		vegan=json.load(json_file)
		tweetCount.append(len(vegan['statuses']))
	 	VeganData.append(vegan['statuses'])
#print fileCount	 	
tweetsVegan=[]
for i in range(fileCount):
	for j in range(tweetCount[i]):
		tweetsVegan.append(VeganData[i][j]['text'])
print len(tweetsVegan)


tweetsVegan,sentences=cleaning(tweetsVegan)
#print tweetsVegan[:200]

#print unigrams[:2000]
#print unigrams
#print cluster(unigrams[:10000])
#print Word2Vector(sentences,tweetsVegan[:1000])
wordsCounter,words= Dict2(tweetsVegan)
print len(words)
print wordsCounter
#bigrams=generate_bigrams(words)
#print Counter(bigrams)
#print bigrams[:100]

#print wordsCounter
#print unigrams[:100]
#print lda1(words)
wc(words)