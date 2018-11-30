import os
import gensim
import nltk
from pyTweetCleaner import TweetCleaner
import json
import ast
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
# import pyLDAvis

BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, 'data/input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/output')


class Preprocessor:
	"""
	Class to perform preprocessing operations needed before training the model.
	"""
	def __init__(self, load_model=False):
		self.stopwords = set()
		self.doc_clean = []
		self.stopwords = set(stopwords.words('english'))
		self.exclude = set(string.punctuation)
		self.lemma = WordNetLemmatizer()
		# self.model_name = 'GoogleNews-vectors-negative300.bin.gz'
		# if load_model:
		#   self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_name, binary=True)

	def extract_tweets_from_file(self, filepath):
		with open(filepath) as f:
			tweets = f.readlines()
		return tweets

	def clean_data(self):
		tc = TweetCleaner(remove_stop_words=False, remove_retweets=False)
		input_files = os.listdir(INPUT_DIR)

		for filename in input_files:
			tc.clean_tweets(input_file=os.path.join(INPUT_DIR, filename), \
				output_file=os.path.join(OUTPUT_DIR, filename))
		
		print("Done")

	def clean(self,doc):
		"""

		"""
		stop_free = " ".join([i for i in doc.lower().split() if i not in self.stopwords])
		punc_free = ''.join(ch for ch in stop_free if ch not in self.exclude)
		normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
		return normalized


	def get_text(self):
		tweets = []
		output_files = os.listdir(OUTPUT_DIR)

		for filename in output_files:
			filepath = os.path.join(OUTPUT_DIR, filename)
			data = open(filepath).readlines()
			for row in data:
				tweet = json.loads(row)
				tweets.append(tweet['text'])
				# print(tweet['text'])
				
		print(len(tweets))
		return tweets

	def clustering(self,tweets):

		for doc in tweets:
			print(doc)
			self.doc_clean.append(self.clean((doc)).split())

		self.dictionary = corpora.Dictionary(self.doc_clean)
		self.doc_term_matrix = [self.dictionary.doc2bow(tweet) for tweet in self.doc_clean] 
		Lda = gensim.models.ldamodel.LdaModel
		ldamodel = Lda(self.doc_term_matrix, num_topics=9, id2word = self.dictionary ,passes=50)
		print(ldamodel.print_topics(num_topics=9, num_words=15))
		# pyLDAvis.gensim.prepare(ldamodel, self.doc_term_matrix, self.dictionary)
		

def generate_word_cloud():
	filename = 'wordcloud_input.txt'
	data = []
	with open(filename, 'r') as f:
		data = f.read()
		data = ast.literal_eval(data)
	for row in data:
		freq_dict = {}
		freq_data = row[1].split('+')
		num = len(freq_data)
		for i in range(num):
			freq_data[i] = freq_data[i].strip().replace('"','')
			value, key = freq_data[i].split('*')
			freq_dict[key] = float(value)
		print(freq_dict)
		wordcloud = WordCloud().fit_words(freq_dict)
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis('off')
		plt.show()


if __name__ == '__main__':
	
	#process = Preprocessor()
	#tweets = process.get_text()
	#process.clustering(tweets)
	generate_word_cloud()
