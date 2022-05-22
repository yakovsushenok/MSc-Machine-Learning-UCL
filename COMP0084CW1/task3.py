import itertools
import math
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
import json
import re
from nltk.stem import WordNetLemmatizer

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

with open("inv_indx.json", "r") as json_file:
    inv_indx = json.load(json_file)

candidate_passageDF = pd.read_table('candidate-passages-top1000.tsv', header = None)
candidate_passageDF.columns = ['qid', 'pid', 'query', 'passage']
candidate_passageDF["passage"] = candidate_passageDF["passage"].apply(text_preprocess) # Now let's preprocess the data the same way we processed the vocabulary
candidate_passageDF_noDupPas = candidate_passageDF.drop_duplicates(subset=['pid'], keep='first')
candidate_passageDF_noDupPas.drop(['qid', 'query'], axis = 1, inplace = True) 
candidate_passageDF_noDupPas = candidate_passageDF_noDupPas.reset_index()

# We will use this code block below to get the words which are present in the queries 
candidate_passageDF_noDupQue = candidate_passageDF.drop_duplicates(subset=['qid'], keep='first')
candidate_passageDF_noDupQue['query'] = candidate_passageDF_noDupQue['query'].apply(text_preprocess)
queryWordDict = defaultdict(list)
for query in candidate_passageDF_noDupQue["query"]:
  for word in query:
    queryWordDict[word] = defaultdict(list)
queryWordList = []
for k, v in queryWordDict.items():
  queryWordList.append(k)

TOTAL_NUM_OF_PASSAGES = 182469
TF_IDF =  defaultdict(list)
idf_query = {}
IDFBM25 = {}
idf_queryBM25 = {}
tf_in_passage = defaultdict(list)

# Building the TF-IDF representations for the passage (also IDF for the query words)
for idx, text in enumerate(candidate_passageDF_noDupPas["passage"]):
  
  tfidf_passage = {}
  tf_loop_passage = {}
  
  for word in text:
    
    tf = text.count(word)/float(len(text)) # The inverted index built in task 2 can be used here!!! But i think this way is faster. Also, this can be in a sense called an inverted index
    idf = math.log10(TOTAL_NUM_OF_PASSAGES/len(inv_indx[word]))
    tf_loop_passage[word] = tf
    idfBM25 = math.log10((TOTAL_NUM_OF_PASSAGES-len(inv_indx[word])+0.5)/(len(inv_indx[word])+0.5))
    
    if word in queryWordList:
      idf_query[word] = idf
      idf_queryBM25[word] = idfBM25
    
    tfidf_passage[word] = tf*idf
    IDFBM25[word] = idfBM25
  
  tf_in_passage[candidate_passageDF_noDupPas.loc[idx,'pid']].append(tf_loop_passage)
  TF_IDF[candidate_passageDF_noDupPas.loc[idx, "pid"]].append(tfidf_passage)




# For those words which are in the queries but are not in the documents we set their IDF to log(N) since this is done in some cases 
for word in queryWordList:
  if word not in list(idf_query.keys()):
    idf_query[word] = math.log10(TOTAL_NUM_OF_PASSAGES/1)
    idf_queryBM25[word] = (TOTAL_NUM_OF_PASSAGES + 0.5)/ 0.5

# Building the TF-IDF representations for the queries
TF_IDF_QUE = defaultdict(list)
tf_in_query = defaultdict(list)

candidate_passageDF_noDupQue = candidate_passageDF.drop_duplicates(subset=['qid'], keep='first')
candidate_passageDF_noDupQue['query'] = candidate_passageDF_noDupQue['query'].apply(text_preprocess)
candidate_passageDF_noDupQue = candidate_passageDF_noDupQue.reset_index()

for idx, query in enumerate(candidate_passageDF_noDupQue['query']):
  
  idf_q = defaultdict(list)
  tf_loop_que = {}
  
  for word in query:
    
    tf = query.count(word)/float(len(query))  # Note that the inverted index can be used here!! However I think that it's faster this way
    if tf == 0: # Some queries after being processed appear to have no words appearing in them, so we set the term frequency to 1 (arbitrary number)
      tf = 1
    tf_loop_que[word] = tf
    idf_q[word] = tf*idf_query[word]
  
  tf_in_query[candidate_passageDF_noDupQue.loc[idx,'qid']].append(tf_loop_que)
  TF_IDF_QUE[candidate_passageDF_noDupQue.loc[idx,'qid']].append(idf_q)

t1 = time.time()
total = t1-t0

print(f'total time taken for Cosine and BM25 data prep: {total} s')  # 72 sec

t0 = time.time()
# Creating the cosine similarity scores for each query-passage pair
def docAndQueryLength(queryDict, DocDict):
  common_words = list(set(list(queryDict.keys())) & set(list(DocDict.keys())))
  queryLength = np.sqrt(sum([queryDict[i]**2 for i in common_words]))
  docLength = np.sqrt(sum([DocDict[i]**2 for i in common_words]))
  return queryLength + docLength
# Calculating average passage length
ln = []
for i in candidate_passageDF_noDupPas['passage']:
  ln.append(len(i))
AVG_PASSAGE_LENGTH = np.mean(ln)
k1 = 1.2
k2 = 100

CosList =  np.zeros(len(candidate_passageDF['qid']))
BM25List =  np.zeros(len(candidate_passageDF['qid']))

for idx, row in enumerate(candidate_passageDF['qid']):
    # Common terms between the query and the passage
    common_terms = list(set(list(TF_IDF[candidate_passageDF.loc[idx, 'pid']][0].keys())) & set(list(TF_IDF_QUE[row][0].keys())))
    # Cosine similarity score
    denominator = docAndQueryLength(TF_IDF[candidate_passageDF.loc[idx, 'pid']][0], TF_IDF_QUE[row][0])
    CosList[idx] = sum([TF_IDF[candidate_passageDF.loc[idx, 'pid']][0][i] * TF_IDF_QUE[row][0][i] for i in common_terms]) / denominator
    # BM25 Score
    KDOC = len(candidate_passageDF.loc[idx, 'passage']) / AVG_PASSAGE_LENGTH
    BM25List[idx] = sum([IDFBM25[i]* ((k1+1)*tf_in_passage[candidate_passageDF.loc[idx, 'pid']][0][i]/(KDOC+tf_in_passage[candidate_passageDF.loc[idx, 'pid']][0][i])) * ((k2+1)*tf_in_query[candidate_passageDF.loc[idx,'qid']][0][i]/(k2+tf_in_query[candidate_passageDF.loc[idx,'qid']][0][i])) for i in common_terms])                                                 

candidate_passageDF['CosSim'] = pd.Series(CosList)
candidate_passageDF['BM25'] = pd.Series(BM25List)

# Getting the best 100 passages from each 1000 passages (or less) associated to each query
testDF = pd.read_table('test-queries.tsv', header = None)
testDF.columns = ['qid', 'query']

DFLIST = []
DFLISTBM25 = []
for idx, qid in enumerate(testDF['qid']):

  subsetQueDF = candidate_passageDF[candidate_passageDF['qid'] == qid]
  subsetQueDFCos = subsetQueDF.sort_values('CosSim',ascending=False)[:100] # Cosine
  subsetQueDFCos = subsetQueDFCos[['qid','pid','CosSim']]
  subsetQueDFCos = subsetQueDFCos.reset_index(drop=True)
  DFLIST.append(subsetQueDFCos)

  subsetQueDFBM25 = subsetQueDF.sort_values('BM25',ascending=False)[:100] # BM25
  subsetQueDFBM25 = subsetQueDFBM25[['qid','pid','BM25']]
  subsetQueDFBM25 = subsetQueDFBM25.reset_index(drop=True)
  DFLISTBM25.append(subsetQueDFBM25)

dfCos = pd.DataFrame(DFLIST[0])
for i in range(1,len(DFLIST)):
  df1 = pd.DataFrame(DFLIST[i])
  dfCos = pd.concat([dfCos, df1])

dfBM25 = pd.DataFrame(DFLISTBM25[0])
for i in range(1,len(DFLISTBM25)):
  df1 = pd.DataFrame(DFLISTBM25[i])
  dfBM25 = pd.concat([dfBM25, df1])

t2 = time.time()
total = t2-t0

print(f'total time taken for Cosin similarity calculation: {total} s')  # Total of 110 sec