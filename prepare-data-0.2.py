#!/usr/local/bin/python
import numpy as np
import pandas
import h5py
from pandas import HDFStore, DataFrame


#Loading the essays in the train set
def loadData(filename):
	scores = np.loadtxt(filename, delimiter='\t', usecols=[1], dtype=np.float)
	essays = np.loadtxt(filename, delimiter='\t', usecols=[2], dtype=np.str)
	return(essays, scores)

def generateVocabulary(train_data):
	#reading and getting a list of unique tokens
	wordslist = set() #initializing the array
        for line in train_data:
	    words = line.split(" ")
    	    for word in words:
                word = word.lower().replace("\n", "")
                if(word not in wordslist):
                    #wordslist contains the vocabulary of the training set
                    wordslist.add(word)
	return wordslist

#Loading scores of the train data
def getScores(filename):
	train_scores = np.loadtxt(filename, dtype = np.int)
        return train_scores

#Loading glove, reducing its size to that of the vocabulary
def loadingGlove(filename, outfile):
	glovevecs = np.loadtxt(filename, delimiter=' ', dtype=np.float, usecols=range(1,301)) #exclude the first column
	glovewords = np.loadtxt(filename, delimiter=' ', dtype=np.str, usecols=range(0,1)) 
	#selecting a subset of words from the glovevec object
	glovemat = []
	selectedwords = []
	for i in range(0, len(glovewords)): 
    		if glovewords[i] in wordslist:
        		glovemat.append(glovevecs[i])
        		selectedwords.append(glovewords[i])
	glovemat = np.array(glovemat)#contains the float values of the word vectors
	#print("Size of the glovevec matrix:")
	#print(len(glovemat))
	#writing out the subset of the the glove vector matrix
	store = h5py.File(outfile, "w")
	store.create_dataset("glovevec", data=glovemat)
	store.close()
	return selectedwords

#Generating key-index pairs
def generateWordIndexPairs(selectedwords):
	wordIndex = np.array
	wordIndex = np.vstack((selectedwords, range(1, (len(selectedwords)+1)))).T 
	#generating the dictionary
	wordIndexDict = dict(list(wordIndex))
	return wordIndexDict

#Getting indices of the vocabulary terms, and creating vectors of indices for each of the data points
def getVectorsOfIndices(essays, wordIndexDict):
	essays_index = []
	for essay in essays:
    	    tokens = essay.split(" ")
            temp = []
            for token in tokens:
                if wordIndexDict.has_key(token): #if token is present in the dictionary
                    temp.append(int(wordIndexDict[token]))
            essays_index.append(temp)
	return essays_index

def writeVectorsOfIndices(indices, scores, filename):
	store = h5py.File(filename, "w")
	df = DataFrame(indices)
	df.fillna(-1, inplace =True)
	store.create_dataset("index", data=df, dtype=np.int) 
	store.create_dataset("target", data=scores, dtype=np.int) 
	store.close()

#Load train file
output = loadData("../Colorado/Operational Data/trainValidTestSets/COSC120160-train.csv")
train_essays = output[0] 
train_scores = output[1] 
print(train_scores)
#Load validation file
output = loadData("../Colorado/Operational Data/trainValidTestSets/COSC120160-valid.csv")
validation_essays = output[0] 
validation_scores = output[1] 
#Load test file
output = loadData("../Colorado/Operational Data/trainValidTestSets/COSC120160-test.csv")
test_essays = output[0] 
test_scores = output[1]
#Using train data to get vocabulary
wordslist = generateVocabulary(train_essays)
print(len(wordslist))

#Load glove vectors
wordOrderInGlove = loadingGlove("../Colorado/Operational Data/Spring2015Prompts/GloveAveragingFeats/glove.6B.300d.txt", "data/glovevec.COSC120160.h5")
# print(wordOrderInGlove)
# print(len(wordOrderInGlove))
# 
# #Get Word-Index dictionary
wordIndexDict = generateWordIndexPairs(wordOrderInGlove)
# print(wordIndexDict)
# 
# #Get index vectors for the train and test sets
train_essay_index = getVectorsOfIndices(train_essays, wordIndexDict)
validation_essay_index = getVectorsOfIndices(validation_essays, wordIndexDict)
test_essay_index = getVectorsOfIndices(test_essays, wordIndexDict)
# 
# #Writing out the index files
writeVectorsOfIndices(train_essay_index, train_scores, "data/COSC120160_train.h5")
writeVectorsOfIndices(validation_essay_index, validation_scores, "data/COSC120160_validation.h5")
writeVectorsOfIndices(test_essay_index, test_scores, "data/COSC120160_test.h5")

