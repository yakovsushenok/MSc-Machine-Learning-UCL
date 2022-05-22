import numpy as np 
import nltk
nltk.download('punkt')
nltk.download("stopwords")
from nltk.corpus import PlaintextCorpusReader
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.probability import FreqDist
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import time
import re
from nltk.stem import WordNetLemmatizer
import json

def text_preprocess(text, dicti = False):
    stop = set(stopwords.words("english")) # len(stop) = 179
    text = re.sub("[^a-zA-Z]+", r' ',text)
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    text = " ".join(text)
    text = text.lower()
    words = text.split()
    wordnet_lemmatizer = WordNetLemmatizer()
    word_list = []
    if dicti == False:
      for w in words:
          woLem = wordnet_lemmatizer.lemmatize(w)
          word_list.append(woLem)
      return word_list
    else:
      fdist = FreqDist()
      for w in words:
          woLem = wordnet_lemmatizer.lemmatize(w)
          word_list.append(woLem)
          fdist[woLem] += 1
      return fdist, word_list


t0 = time.time()

file_content = open("passage-collection.txt").read()
fdist, file_content = text_preprocess(file_content, dicti = True)   
print(f" Number of distinct words (Vocabulary size): {len(fdist)}") 
candidate_passageDF = pd.read_table('candidate-passages-top1000.tsv', header = None)
candidate_passageDF.columns = ['qid', 'pid', 'query', 'passage']
# Inverted index for the passages in candidate-passages-top1000.tsv with the words from the vocabulary from task 1.
# First let's drop all duplicates of passages
candidate_passageDF["passage"] = candidate_passageDF["passage"].apply(text_preprocess) # Now let's preprocess the data the same way we processed the vocabulary
candidate_passageDF_noDupPas = candidate_passageDF.drop_duplicates(subset=['pid'], keep='first')
candidate_passageDF_noDupPas.drop(['qid', 'query'], axis = 1, inplace = True) 
candidate_passageDF_noDupPas = candidate_passageDF_noDupPas.reset_index()

# This loop will create a dictionary where the keys are the words and the values are lists which contain the id's of the passages the words appear in 
inv_indx = {}
for pid, pas in zip(candidate_passageDF_noDupPas["pid"],candidate_passageDF_noDupPas["passage"]):
  for word in pas:
    if word not in inv_indx:
      inv_indx[word]={}
    if word in inv_indx: 
      if pid in inv_indx[word]: 
        inv_indx[word][pid] +=1
      else:
        inv_indx[word][pid] = 1


with open('inv_indx.json', 'w') as fp:
    json.dump(inv_indx, fp)




t1 = time.time()
total = t1-t0

print(f'total time taken for task 2: {total} s') # 128