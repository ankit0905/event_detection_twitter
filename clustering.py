from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering

from sklearn.externals import joblib
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, dendrogram
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from sklearn.cluster import AgglomerativeClustering

from load_dataset import load_dataset
from sklearn import metrics

class Kmeans:
	"""

	"""
	def __init__(self,num):
		"""

		"""
		self.num_clusters = num

	def clustering(self,features):
		"""

		"""
		kmeans = KMeans(n_clusters=self.num_clusters)
		kmeans.fit(features)
		# joblib.dump(kmeans,'doc_cluster_kmeans.pkl')
		# kmeans = joblib.load('doc_cluster_kmeans.pkl')
		clusters = kmeans.labels_.tolist()
		return clusters
		# ct = {}

		# for label in clusters:
		# 	if label in ct.keys():
		# 		ct[label] = ct[label] + 1
		# 	else:
		# 		ct[label] = 1

		# print("######from kmeans clustering ##########")
		# print(ct)

class WardClustering:
	"""

	"""
	def __init__(self,feature_matrix):
		"""

		"""
		self.features = feature_matrix

	def clustering(self):
		"""

		"""
		linkage_matrix = ward(self.features)
		fig, ax = plt.subplots(figsize=(15, 20))
		ax = dendrogram(linkage_matrix, orientation="right")
		plt.tick_params(\
			axis = 'x',
			which = 'both',
			bottom = 'off',
			top = 'off',
			labelbottom = 'off')

		plt.tight_layout()
		plt.savefig('ward_clusters.png',dpi=200)





def compare_labels(labels, y):
	ct1 = ct2 = 0
	num = len(y)
	for i in range(num):
		for j in range(num):
			if labels[i] == labels[j]:
				if y[i] == y[j]:
					ct1 += 1
				else:
					ct2 += 1
			else:
				if y[i] == y[j]:
					ct2 += 1
				else:
					ct1 += 1
	return (ct1 / (ct1 + ct2))

def get_score(labels_true, labels_pred):

	arc = metrics.adjusted_rand_score(labels_true, labels_pred)
	mif = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
	homogenity = metrics.homogeneity_score(labels_true, labels_pred) 
	completeness = metrics.completeness_score(labels_true, labels_pred)
	fms = metrics.fowlkes_mallows_score(labels_true, labels_pred)

	print(arc)
	print(mif)
	print(homogenity)
	print(completeness)
	print(fms)




if __name__ == '__main__':
	X, y, num_classes = load_dataset()
	print("num_classes" + str(num_classes))

	print("################ original cnt ##############")
	# print(org)      

	# kmeans = Kmeans(num_classes)
	# labels = kmeans.clustering(X)
	# get_score(y,labels)
	# wardclustering = WardClustering(X)
	# wardclustering.clustering()

	# brc = Birch(branching_factor=50, n_clusters=num_classes, threshold=0.5,compute_labels=True)
	# labels = brc.fit_predict(X)
	# get_score(y,labels)

	# cluster = SpectralClustering(n_clusters=num_classes-2,assign_labels="discretize",random_state=0).fit(X)
	# labels = cluster.labels_
	# get_score(y,labels)


	cluster = AgglomerativeClustering(n_clusters=num_classes, affinity='cosine', linkage='complete')
	labels = cluster.fit_predict(X)
	get_score(y,labels)
	# equal = 0
	# diff = 0

	# print(type(labels))

	# print(compare_labels(labels,y))

	# Lda = gensim.models.ldamodel.LdaModel
	# ldamodel = Lda(self.doc_term_matrix, num_topics=2, id2word = self.dictionary ,passes=50)
	# print(ldamodel.print_topics(num_topics=2, num_words=10))