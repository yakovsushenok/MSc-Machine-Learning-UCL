import numpy as np 
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download("stopwords")
from nltk.corpus import PlaintextCorpusReader
from collections import defaultdict
from nltk import ngrams
from nltk.tokenize import word_tokenize
import string
import math
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import re

np.random.seed(42)
train_DF = pd.read_table('./train_data.tsv', header = 0)
# sampling 25% from the data
train_DF = train_DF.sample(frac=0.25) 
val_DF = pd.read_table('./validation_data.tsv', header = 0)


###################################################################################################################
#                                                                                                                 #
#                                         Code from assignment 1                                                  #
#                                                                                                                 #
###################################################################################################################                                                                                   




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

val_DF['queries'] = val_DF['queries'].apply(text_preprocess)
val_DF['passage'] = val_DF['passage'].apply(text_preprocess)

val_DF_noDupPas = val_DF.drop_duplicates(subset=['pid'], keep='first')
val_DF_noDupPas = val_DF_noDupPas.reset_index()

val_DF_noDupQue = val_DF.drop_duplicates(subset=['qid'], keep='first')
# This loop will create a dictionary where the keys are the words and the values are lists (with duplicates) which contain the id's of the passages the words appear in 
inv_indx = {}
for pid, pas in zip(val_DF_noDupPas["pid"],val_DF_noDupPas["passage"]):
  for word in pas:
    if word not in inv_indx:
      inv_indx[word]={}
    if word in inv_indx: 
      if pid in inv_indx[word]: 
        inv_indx[word][pid] +=1
      else:
        inv_indx[word][pid] = 1

# We will use this code block below to get the words which are present in the queries 1103039 b4, 1148 after
queryWordDict = defaultdict(list)
for query in val_DF_noDupQue["queries"]:
  for word in query:
    queryWordDict[word] = defaultdict(list)
queryWordList = []
for k, v in queryWordDict.items():
  queryWordList.append(k)

TOTAL_NUM_OF_PASSAGES = len(val_DF_noDupPas) # 955211
TF_IDF =  defaultdict(list)
idf_query = {}
IDFBM25 = {}
idf_queryBM25 = {}
tf_in_passage = defaultdict(list)
# Building the TF-IDF representations for the passage (also IDF for the query words)
for idx, text in enumerate(val_DF_noDupPas["passage"]):
  
  tfidf_passage = {}
  tf_loop_passage = {}
  
  for word in text:
    
    tf = text.count(word)/float(len(text)) 
    idf = math.log10(TOTAL_NUM_OF_PASSAGES/len(inv_indx[word]))
    tf_loop_passage[word] = tf
    idfBM25 = math.log10((TOTAL_NUM_OF_PASSAGES-len(inv_indx[word])+0.5)/(len(inv_indx[word])+0.5))
    
    if word in queryWordList:
      idf_query[word] = idf
      idf_queryBM25[word] = idfBM25
    
    tfidf_passage[word] = tf*idf
    IDFBM25[word] = idfBM25
  tf_in_passage[val_DF_noDupPas.loc[idx,'pid']].append(tf_loop_passage)
  TF_IDF[val_DF_noDupPas.loc[idx, "pid"]].append(tfidf_passage)

# For those words which are in the queries but are not in the documents we set their IDF to log(N) since this is done in some cases 
for word in queryWordList:
  if word not in list(idf_query.keys()):
    idf_query[word] = math.log10(TOTAL_NUM_OF_PASSAGES/1)
    idf_queryBM25[word] = (TOTAL_NUM_OF_PASSAGES + 0.5)/ 0.5

# Building the TF-IDF representations for the queries
TF_IDF_QUE = defaultdict(list)
tf_in_query = defaultdict(list)

val_DF_noDupQue = val_DF_noDupQue.reset_index()

for idx, query in enumerate(val_DF_noDupQue['queries']):
  
  idf_q = defaultdict(list)
  tf_loop_que = {}
  
  for word in query:
    
    tf = query.count(word)/float(len(query))  # Note that the inverted index can be used here!! However I think that it's faster this way
    if tf == 0: # Some queries after being processed appear to have no words appearing in them, so we set the term frequency to 1 (arbitrary number)
      tf = 1
    tf_loop_que[word] = tf
    idf_q[word] = tf*idf_query[word]
  
  tf_in_query[val_DF_noDupQue.loc[idx,'qid']].append(tf_loop_que)
  TF_IDF_QUE[val_DF_noDupQue.loc[idx,'qid']].append(idf_q)


# Calculating average passage length
ln = []
for i in val_DF_noDupPas['passage']:
  ln.append(len(i))
AVG_PASSAGE_LENGTH = np.mean(ln)
k1 = 1.2
k2 = 100

BM25List =  np.zeros(len(val_DF['qid']))
for idx, row in enumerate(val_DF['qid']):
    # Common terms between the query and the passage
    common_terms = list(set(list(TF_IDF[val_DF.loc[idx, 'pid']][0].keys())) & set(list(TF_IDF_QUE[row][0].keys())))
    # BM25 Score
    KDOC = len(val_DF.loc[idx, 'passage']) / AVG_PASSAGE_LENGTH
    BM25List[idx] = sum([IDFBM25[i]* ((k1+1)*tf_in_passage[val_DF.loc[idx, 'pid']][0][i]/(KDOC+tf_in_passage[val_DF.loc[idx, 'pid']][0][i])) * ((k2+1)*tf_in_query[val_DF.loc[idx,'qid']][0][i]/(k2+tf_in_query[val_DF.loc[idx,'qid']][0][i])) for i in common_terms])  

val_DF['BM25'] = pd.Series(BM25List)
###################################################################################################################
#                                                                                                                 #
#                                         Code from assignment 2 task 1                                           #
#                                                                                                                 #
###################################################################################################################  

# first we will calculate the mean average precision
averagePrecisionList = []
for idx, qid in enumerate(val_DF_noDupQue['qid']):
  subsetQueDF = val_DF[val_DF['qid'] == qid]
  subsetQueDFBM25 = subsetQueDF.sort_values('BM25',ascending=False) # BM25
  subsetQueDFBM25[:100]
  relevance = np.array(subsetQueDFBM25['relevancy'])
  cs = np.cumsum(relevance)
  div = np.arange(1,len(relevance)+1)
  precisionList = np.divide(cs, div)
  averagePrecisionList.append(np.mean(precisionList))

AP = np.mean(averagePrecisionList)
print(AP)

ndcglist = []
# now we will calculate NDCG
for idx, qid in enumerate(val_DF_noDupQue['qid']):
  subsetQueDF = val_DF[val_DF['qid'] == qid]
  subsetQueDFBM25 = subsetQueDF.sort_values('BM25',ascending=False)
  subsetQueDFBM25 = subsetQueDFBM25[:100]
  relevance = np.array(subsetQueDFBM25['relevancy'])
  gain = 2**relevance - 1
  denom = np.log2(np.arange(1,len(relevance)+1))+1
  dcgelems = np.divide(gain,denom)
  dcg = np.sum(dcgelems)
  # calculating ideal dcg
  irel = np.sort(relevance)[::-1]
  gain = 2**irel - 1
  denom = np.log2(np.arange(1,len(irel)+1))+1
  dcgelems = np.divide(gain,denom)
  idcg = np.sum(dcgelems)
  #if idcg == 0:
   # continue
  NDCG = dcg/(idcg+0.1)
  ndcglist.append(NDCG)
meanNDCG = np.mean(ndcglist)
print(meanNDCG)