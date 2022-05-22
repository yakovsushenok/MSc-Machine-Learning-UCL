import numpy as np 
import nltk
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



queries = train_DF_noDupQue['queries'].values.tolist()
passages = train_DF_noDupPas['passage'].values.tolist()
corpus = queries+passages
token_corp = [nltk.word_tokenize(sent) for sent in corpus]

model = gensim.models.Word2Vec(token_corp, min_count=1, size = 32)

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

train_DF['Cosine Similarity'] = [np.dot(x,y)/(sum(x**2)*(sum(y**2))) for x,y in zip(train_DF['passage embedding'],train_DF['query embedding'])]

import xgboost as xgb
X_train = train_DF.loc[:, train_DF.columns.isin(['Cosine Similarity'])]
y_train = train_DF.loc[:, train_DF.columns.isin(['relevancy'])]
groups = train_DF.groupby('qid').size().to_frame('size')['size'].to_numpy()
# n_estimator_list = [40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150, 160]   
# colsample_bytree_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# eta_list = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 ]
# max_depth_list = [2,4,6,8,10,12,14,16,18,20]
# lr_list = [0.001,0.01,0.1,0.5]
n_estimator_list = [64]   
colsample_bytree_list = [0.86]
eta_list = [0.27]
max_depth_list = [14]
lr_list = [0.01]

lmdf = []
for n in n_estimator_list:
  for c in colsample_bytree_list:
    for e in eta_list:
     for m in max_depth_list:
       for l in lr_list:
          model = xgb.XGBRanker(  
              objective='rank:pairwise',
              random_state=42, 
              learning_rate=l,
              colsample_bytree=c, 
              eta=e, 
              max_depth=m, 
              n_estimators=n, 
              subsample=0.75 
              )

          model.fit(X_train, y_train, group=groups, verbose=True)

          def predict(model, df):
              return model.predict(df.loc[:, df.columns.isin(['Cosine Similarity'])])
            
          predictions = (val_DF.groupby('qid')
                        .apply(lambda x: predict(model, x)))


          val_DF_grqid =  val_DF.sort_values(by=['qid'], ascending=True)
          rank_score = np.concatenate([i for i in predictions])
          val_DF_grqid['lm score'] = rank_score
          val_DFLM = val_DF_grqid.groupby('qid', as_index= False).apply(lambda x: x.sort_values('lm score', ascending = False))
          val_DFLM = val_DFLM.reset_index()[['qid','pid','relevancy','lm score']]


          # evaluation
          val_DF_noDupQue = val_DFLM.drop_duplicates(subset=['qid'], keep='first')
          # first we will calculate the mean average precision
          averagePrecisionList = []
          for idx, qid in enumerate(val_DF_noDupQue['qid']):
            subsetQueDF = val_DFLM[val_DF['qid'] == qid]
            subsetQueDFBM25 = subsetQueDF.sort_values('lm score',ascending=False) # BM25
            relevance = np.array(subsetQueDFBM25['relevancy'])
            cs = np.cumsum(relevance)
            div = np.arange(1,len(relevance)+1)
            precisionList = np.divide(cs, div)
            averagePrecisionList.append(np.mean(precisionList))

          AP = np.mean(averagePrecisionList)

          ndcglist = []
          # now we will calculate NDCG
          for idx, qid in enumerate(val_DF_noDupQue['qid']):
            subsetQueDF = val_DFLM[val_DFLM['qid'] == qid]
            subsetQueDFBM25 = subsetQueDF.sort_values('lm score',ascending=False)
            subsetQueDF = subsetQueDFBM25[:100]
            rank = np.arange(1,len(subsetQueDF)+1)
            ff = ["LM"]*len(subsetQueDF)
            a1 = ['A1']*len(subsetQueDF)
            subsetQueDF['a1'] = a1
            subsetQueDF['rank'] = rank
            subsetQueDF['algo'] = ff
            lmdf.append(subsetQueDF)
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
            NDCG = dcg/(idcg+1)
            ndcglist.append(NDCG)
          meanNDCG = np.mean(ndcglist)
          print(f"n: {n}, c: {c}, e: {e}, m: {m}, l: {l}")
          print(f"Mean avg precision: {AP}, Mean ndgc {meanNDCG}, mean metric= {(AP+meanNDCG)/2}") 

dflm = pd.DataFrame(lmdf[0])
for i in range(1,len(lmdf)):
  df1 = pd.DataFrame(lmdf[i])
  dflm = pd.concat([dflm, df1])

colist = ['qid', 'a1', 'pid','rank','lm score', 'algo']
# <qid1 A1 pid1 rank1 score1 algoname2
dflm= dflm[colist]
dflm.to_csv('lm.csv')
