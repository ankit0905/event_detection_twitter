'''
	Steps to be followed
	1. Data Extraction
	2. Data Cleaning & Preprocessing
	3. Use of Word2Vec Model to generate vectors
	4. Creation of Training and Test Data (Split)
	5. Model Training (Classifiers: ANN, SVM, ...) - save model
	6. Calculation of accuracy on test data
'''

from helper import Preprocessor

if __name__ == '__main__':
	preprocessor = Preprocessor()
	preprocessor.clean_data() # run once