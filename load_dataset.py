import os
import numpy as np

BASE_DIR = 'C:/Users/ankit/Documents/project_/event_detection_twitter/data/average_vectors'

def load_dataset():
	label = 0
	X = []
	Y = []
	for dir in os.listdir(BASE_DIR):
		size = 0
		tmp_dir = os.path.join(BASE_DIR, dir)
		for filename in os.listdir(tmp_dir):
			vecs = np.load(os.path.join(tmp_dir, filename))
			X.extend(vecs)
			size += len(vecs)
		Y.extend([label]*size)
		label += 1
	X = np.array(X)
	Y = np.array(Y)
	print(X.shape, Y.shape)
	return X, Y, label

if __name__ == '__main__':
	load_dataset()