import glob
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
#Data retrieved from Twitter
def cleaning(tweets):

	additional_stopwords=['amp', '#vegan', '@', 'rt','vegan','#','veganhour','govegan','plantbased']
	stop=set(stopwords.words('english'))
	#target=open('tweets.txt','w')
	#Remove repeated tweets
	tweets=list(set(tweets))
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
		tweets[i]=re.sub(r'[/.!$%^&*():,-;=?~]', ' ', tweets[i])
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
	
	return tweets
#sample=tweets[0]
def analysis(tweets):
	unigrams=[]
	for i in range(len(tweets)):
		words=tweets[i].split()
		unigrams.append(words)
	unigrams=list(itertools.chain(*unigrams))

	bigrams=[]
	for i in range(len(tweets)):
		pairs = [b for b in zip(tweets[i].split(" ")[:-1], tweets[i].split(" ")[1:])]
		for pair in pairs:
			bigram=''.join(pair)
			bigrams.append(bigram)		
	#bigrams=list(itertools.chain(*bigrams))
	return unigrams

def wc(unigrams):
	wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(unigrams))
	plt.figure(figsize=(15,8))
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

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
print fileCount	 	
tweetsVegan=[]
for i in range(fileCount):
	for j in range(tweetCount[i]):
		tweetsVegan.append(VeganData[i][j]['text'])
	#pprint (tweetsVegan[0])
print len(tweetsVegan)	
tweetsVegan=cleaning(tweetsVegan)
print len(tweetsVegan)
unigrams=analysis(tweetsVegan)
#print unigrams
print cluster(unigrams[:10000])

wc(unigrams)