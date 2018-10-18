import os
import gensim

from pyTweetCleaner import TweetCleaner

BASE_DIR = 'C:/Users/ankit/Documents/project_/event_detection_twitter'
INPUT_DIR = os.path.join(BASE_DIR, 'data/input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/output')


class Preprocessor:
	"""
	Class to perform preprocessing operations needed before training the model.
	"""
	def __init__(self):
		self.stopwords = set()

	def extract_tweets_from_file(self, filepath):
		with open(filepath) as f:
			tweets = f.readlines()
		return tweets

	def clean_data(self):
		tc = TweetCleaner(remove_stop_words=True, remove_retweets=False)
		input_files = os.listdir(INPUT_DIR)

		for filename in input_files:
			tc.clean_tweets(input_file=os.path.join(INPUT_DIR, filename), \
				output_file=os.path.join(OUTPUT_DIR, filename))
		
		print("Done")

	def get_word2vec_vectors(self, model_file):
		# model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=True)
		pass