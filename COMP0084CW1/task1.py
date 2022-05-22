import numpy as np 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import PlaintextCorpusReader
from nltk import ngrams
from nltk.tokenize import word_tokenize
import string
from nltk.probability import FreqDist
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import re
import json

def text_preprocess(text, dicti = False):
    text = re.sub("[^a-zA-Z]+", r' ',text)
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

file_content = open("passage-collection.txt").read()
fdist, file_content = text_preprocess(file_content, dicti = True)

with open('fdist.json', 'w') as fp:
    json.dump(fdist, fp)

sort_by_value = dict(sorted(fdist.items(), key = lambda item: item[1],reverse=True))    # Here we will see the frequency of each word (and punctuation symbol) in descending order

print(f" Number of distinct words (Vocabulary size): {len(fdist)}") 
# Normalizing the frequencies
sum_of_values = sum(sort_by_value.values())
norm_fdist =  {key: sort_by_value[key]/sum_of_values for key in sort_by_value}
# List of the keys
keysList = list(norm_fdist.keys())
# List of the values
valuesList = list(norm_fdist.values())


####################################################################################################################################


# Setting up a dataframe 
df = pd.DataFrame({'Word': keysList, 'Frequency': valuesList})
df['Rank'] = df.index + 1
df["Rank*Frequency"] = df['Rank'] * df['Frequency']
df['LogFrequency'] = np.log(df["Frequency"])
df["LogRank"] = np.log(df["Rank"])
# Getting the Zipf's Law line
Nomalizing_Constant = np.sum(1/df["Rank"])
df["ZipfsLaw"] = 1/ (df["Rank"] *  Nomalizing_Constant)
df["LogZipfsLaw"] = np.log(df["ZipfsLaw"])

LogRank, LogFrequency, LogZipfsLaw = df["LogRank"], df['LogFrequency'], df['LogZipfsLaw']
plt.plot(LogRank, LogFrequency, label = "Log(Probability)")
plt.plot(LogRank, LogZipfsLaw, label = "Log(Zipf's Law)")
plt.legend()
plt.show()

print('\n')
C = np.mean(df["Rank*Frequency"])
b = 0.0001
a = 0.000001
df1 = df[(df["Frequency"] <= b) & (df["Frequency"] >= a)]
print(f"Empirical distribution proportion: {np.round(100*len(df1)/len(df),2)}%")

ka = C /a
kb = C / b
print(f"Zipf's law proportion {np.round(100*(ka-kb+1)/len(fdist),2)}%")