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
import pkg_resources
pkg_resources.require("gensim==3.6.0")
import gensim
from gensim import corpora, models, similarities

np.random.seed(42)
train_DF = pd.read_table('./train_data.tsv', header = 0)
# sampling 25% from the data
train_DF = train_DF.sample(frac=0.25) 
val_DF = pd.read_table('./validation_data.tsv', header = 0)

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
      return  ' '.join(word_list)
    else:
      fdist = FreqDist()
      for w in words:
          woLem = wordnet_lemmatizer.lemmatize(w)
          word_list.append(woLem)
          fdist[woLem] += 1
      return fdist, word_list

train_DF_noDupPas = train_DF.drop_duplicates(subset=['pid'], keep='first')
train_DF_noDupPas = train_DF_noDupPas.reset_index()
train_DF_noDupPas['passage'] = train_DF_noDupPas['passage'].apply(text_preprocess)

train_DF_noDupQue = train_DF.drop_duplicates(subset=['qid'], keep='first')
train_DF_noDupQue = train_DF_noDupQue.reset_index()
train_DF_noDupQue['queries'] = train_DF_noDupQue['queries'].apply(text_preprocess)

__requires__= 'gensim==3.6.0'

queries = train_DF_noDupQue['queries'].values.tolist()
passages = train_DF_noDupPas['passage'].values.tolist()
corpus = queries+passages
token_corp = [nltk.word_tokenize(sent) for sent in corpus]

model = gensim.models.Word2Vec(token_corp, min_count=1, size = 32)
print(gensim.__version__)
#model.save('w2vmodel')

# now that we have all of the terms represented as embeddings, we will compute the query and passage embeddings by averaging the embeddings of the words
def compute_embedding(text):
  embedding = np.zeros(32)
 # text = text.split()
  for word in text:
    if word not in model:
      continue
    else :
      embedding += model[word]
  return embedding/len(text)

train_DF['queries'] = train_DF['queries'].apply(text_preprocess)
train_DF['passage'] = train_DF['passage'].apply(text_preprocess)
train_DF['query embedding'] = train_DF['queries'].apply(compute_embedding)
train_DF['passage embedding'] = train_DF['passage'].apply(compute_embedding)

val_DF['queries'] = val_DF['queries'].apply(text_preprocess)
val_DF['passage'] = val_DF['passage'].apply(text_preprocess)
val_DF['query embedding'] = val_DF['queries'].apply(compute_embedding)
val_DF['passage embedding'] = val_DF['passage'].apply(compute_embedding)
val_DF['Cosine Similarity'] = [np.dot(x,y)/(sum(x**2)*(sum(y**2))) for x,y in zip(val_DF['passage embedding'],val_DF['query embedding'])]

# now we will build a logistic regression model. our model will be based on calculating the cosine similarity of the query and passage and predicting based on that
train_DF['Cosine Similarity'] = [np.dot(x,y)/(sum(x**2)*(sum(y**2))) for x,y in zip(train_DF['passage embedding'],train_DF['query embedding'])]

def predict(X, b0, b1):
    return np.array([1 / (1 + np.exp(-1*b0 + -1*b1*x)) for x in X])
def logReg(X, Y):
    x = X  - X.mean()
    beta0 = 0.01
    beta1 = 0.01
    lr = 0.001
    epochs = 5

    for epoch in range(epochs):
        y_pred = np.array([1 / (1 + np.exp(-1*beta0 + -1*beta1*x)) for x in X])
        derivative0 = -2 * np.nansum((Y - y_pred) * y_pred * (1 - y_pred))  
        derivative1 = -2 * np.nansum(X * (Y - y_pred) * y_pred * (1 - y_pred)) 
        beta0  -=  lr * derivative0
        beta1 -= lr * derivative0
    
    return beta0, beta1

b0, b1 = logReg(np.array(train_DF['Cosine Similarity']), np.array(train_DF['relevancy']))
y_pred = predict(val_DF['Cosine Similarity'], b0, b1)

val_DF['lr score'] = y_pred
val_DF = val_DF.groupby('qid', as_index= False).apply(lambda x: x.sort_values('lr score', ascending = False))
val_DF = val_DF.reset_index()[['qid','relevancy','lr score','pid']]
val_DF_noDupQue = val_DF.drop_duplicates(subset=['qid'], keep='first')
averagePrecisionList = []
for idx, qid in enumerate(val_DF_noDupQue['qid']):
  subsetQueDF = val_DF[val_DF['qid'] == qid]
  subsetQueDF = subsetQueDF.sort_values('lr score',ascending=False) # BM25
  relevance = np.array(subsetQueDF['relevancy'])
  cs = np.cumsum(relevance)
  div = np.arange(1,len(relevance)+1)
  precisionList = np.divide(cs, div)
  averagePrecisionList.append(np.mean(precisionList))

AP = np.mean(averagePrecisionList)
print(AP)



lrdf = []
ndcglist = []
# now we will calculate NDCG
for idx, qid in enumerate(val_DF_noDupQue['qid']):
  subsetQueDF = val_DF[val_DF['qid'] == qid]
  subsetQueDF = subsetQueDF.sort_values('lr score',ascending=False)
  subsetQueDF = subsetQueDF[:100]
  rank = np.arange(1,len(subsetQueDF)+1)
  ff = ["LR"]*len(subsetQueDF)
  a1 = ['A1']*len(subsetQueDF)
  subsetQueDF['a1'] = a1
  subsetQueDF['rank'] = rank
  subsetQueDF['algo'] = ff
  lrdf.append(subsetQueDF)
  relevance = np.array(subsetQueDF['relevancy'])
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
  NDCG = dcg/(idcg+0.1)
  ndcglist.append(NDCG)
meanNDCG = np.mean(ndcglist)
print(meanNDCG)

dflr = pd.DataFrame(lrdf[0])
for i in range(1,len(lrdf)):
  df1 = pd.DataFrame(lrdf[i])
  dflr = pd.concat([dflr, df1])

colist = ['qid', 'a1', 'pid','rank','lr score', 'algo']
# <qid1 A1 pid1 rank1 score1 algoname2
dflr= dflr[colist]
dflr.to_csv('lr.csv')

