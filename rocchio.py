import sys
import os
import nltk
import math
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from collections import Counter
import numpy as np
from string import punctuation

# Hyperparameters
# Manually tuned after testing on three datasets
TERM_SATURATION_CONST = 3
DOC_LENGTH_TUNER = 0.8

class Rocchio():
    # NLTK functions
    default_stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r"\w+")
    wordnet_lemmatizer = WordNetLemmatizer()
    
    def __init__(self, documents, train, centroids = None, base_TFIDF_vector = None, IDF = None, stopwords_path = None, avg_doc_length = None):
        self.documents = documents
        self.train = train
        self.centroids = centroids
        self.base_TFIDF_vector = base_TFIDF_vector
        self.IDF = IDF
        self.stopwords_path = stopwords_path
        self.avg_doc_length = avg_doc_length
        if self.train:
            self.getCentroids()
        else:
            self.predict()

    def getCentroids(self):
        file_term_freqs = {}
        file_labels = {}
        file_unique_tokens = {}
        label_count = defaultdict(int)
        file_length = {}
        with open(self.documents) as fp:
            file_count = 0
            for line in fp:
                path = line.strip().split(' ')[0]
                label = line.strip().split(' ')[1]
                label_count[label] += 1
                tokens = self.getTokens(path)
                file_length[path] = len(tokens)
                file_term_freqs[path] = self.getTermFrequencies(tokens)
                file_labels[path] = label
                # for IDF calculation
                unique_tokens = list(set(tokens))
                file_unique_tokens[path] = list(set(tokens))
                # to save time during debugging
                file_count += 1
        
        # normalize term frequencies
        avg_doc_length = sum(file_length.values()) / float(len(file_length))
        norm_term_freqs = {}
        for path,term_freqs in file_term_freqs.items():
            K = TERM_SATURATION_CONST * (1 - DOC_LENGTH_TUNER +(DOC_LENGTH_TUNER * file_length[path] / avg_doc_length))
            norm_term_freqs[path] = {token: freq/(freq+K) for token,freq in term_freqs.items()}
        self.avg_doc_length = avg_doc_length
                    
        # calculate document frequencies
        doc_freqs = defaultdict(int)
        base_TFIDF_vector = defaultdict(float)
        for tokens in file_unique_tokens.values():
            for token in tokens:
                doc_freqs[token] += 1
                
        # calculate inverse document frequencies
        inv_doc_freqs = {}
        for token,freq in doc_freqs.items():
            idf = math.log(file_count/freq,10)
            inv_doc_freqs[token] = idf
        self.IDF = inv_doc_freqs
            
        # calculate TFIDF for every token in every file
        doc_vectors = {}
        for path,term_freqs in norm_term_freqs.items():
            vector = {}
            for token,term_freq in term_freqs.items():
                # weight = TF * IDF
                weight = term_freq * inv_doc_freqs[token]
                vector[token] = weight
            doc_vectors[path] = vector
        
        # calculate centroids by averaging the vectors of each label
        centroids = {}
        for path,vector in doc_vectors.items():
            label = file_labels[path]
            if label not in centroids:
                centroids[label] = vector
            else:
                centroids[label] = Counter(centroids[label]) + Counter(vector)
        avg_centroids = {}
        for label,centroid in centroids.items():
            avg_centroid = {token:weight/label_count[label] for token,weight in centroid.items()}
            avg_centroids[label] = avg_centroid
        norm_centroids = {label:self.normalize_dict(centroid) for label,centroid in avg_centroids.items()}
        self.centroids = norm_centroids
    
    def getTokens(self,path):
        file_content = open(path).read().lower()
        # remove numbers, punctuation marks, and stopwords
        file_content = ''.join(c for c in file_content if not c.isdigit())
        file_content = ''.join(c for c in file_content if c not in punctuation)
        tokens = self.tokenizer.tokenize(file_content)
        stop_words = self.default_stop_words
        if self.stopwords_path != None:
            stop_words.update(self.getStopwords())
        tokens = [w for w in tokens if not w in stop_words]
        # only consider nouns and verbs
        tokens = [list(x) for x in nltk.pos_tag(tokens) if (x[1][0] == 'N' or x[1][0] == 'V')]
        # lemmatize tokens
        for token in tokens:
            token[0] = self.wordnet_lemmatizer.lemmatize(token[0])
         # append POS to token
        tokens = list(map(lambda x: x[0] + x[1][0], tokens))
        return tokens
    
    # get stopwords from external file
    def getStopwords(self):
        stopwords = []
        with open(self.stopwords_path) as fp:
            for line in fp:
                stopwords.append(line.strip())
        return stopwords

    def getTermFrequencies(self,tokens):
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
        return term_freqs
    
    # adapted from https://stackoverflow.com/questions/63867452/normalization-of-dictionary-values
    def normalize_dict(self,dict):
        X = np.array([val for val in dict.values()])
        norm_2 = np.sqrt((X**2).sum(axis=0))
        norm_dict = {key:dict[key]/norm_2 for key in dict.keys()}
        return norm_dict
    
    def predict(self):
        file_term_freqs = {}
        file_length = {}
        with open(self.documents) as fp:
            file_count = 0
            for path in fp:
                tokens = self.getTokens(path.strip())
                file_length[path] = len(tokens)
                file_term_freqs[path] = self.getTermFrequencies(tokens)
                file_count += 1
                
        # normalize term frequencies
        norm_term_freqs = {}
        for path,term_freqs in file_term_freqs.items():
            K = TERM_SATURATION_CONST * (1 - DOC_LENGTH_TUNER +(DOC_LENGTH_TUNER * file_length[path] / self.avg_doc_length))
            norm_term_freqs[path] = {token: freq/(freq+K) for token,freq in term_freqs.items()}
                
        # calculate TFIDF for every token in every file
        doc_vectors = {}
        for path,term_freqs in norm_term_freqs.items():
            vector = {}
            for token,term_freq in term_freqs.items():
                if token in self.IDF: # ignore tokens that were not in the training set
                    # weight = TF * IDF
                    weight = term_freq * self.IDF[token]
                    vector[token] = weight
            doc_vectors[path.strip()] = vector
            
        # apply cosine similarity to find closest label
        predictions = {}
        for path,TFIDF_vector in doc_vectors.items():
            max = 0
            TFIDF_vector = self.normalize_dict(TFIDF_vector)
            for label,centroid in self.centroids.items():
                cosine_sim = 0
                for token,weight in centroid.items():
                    cosine_sim += weight*TFIDF_vector.get(token,0.0)
                if cosine_sim > max:
                    max = cosine_sim
                    predicted_label = label
            predictions[path] = predicted_label
        self.predictions = predictions
            
        

    

                    
            
            
        
    

