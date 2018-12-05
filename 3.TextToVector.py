
# coding: utf-8

# In[265]:


from ArticlesRep import MeanSimilarityoneindustry,MeanSimilaritytwoindustry #common function

import pandas as pd

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

import pickle

import visdom

from tqdm import tnrange, tqdm_notebook
from tqdm.autonotebook import tqdm
tqdm.pandas()

from time import time,sleep
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
# import visdom
# vis=visdom.Visdom()
# env="TagBased"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline

import re
import sys
from pprint import pprint
#coding:utf-8

#solved chinese display in matplotlib
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' 
mpl.rcParams['axes.unicode_minus'] = False

list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]
element="氫氦鋰鈹硼碳氮氧氟氖鈉鎂鋁矽磷硫氯氬鉀鈣鈧鈦釩鉻錳鐵鈷鎳銅鋅鎵鍺砷硒溴氪銣鍶銀鎘銦錫銻碲碘氙銫鋇鉑金汞鉈鉛鉍釙氡鍅鐳"


# # Outline
# <li>Text to Vector version1
# <li>Text to Vector version2

# ## Text to Vector version1
# <li> Load DataSet_vip_single_industry
# <li> Vectorization Binary one hot of bag word (first 20,000 popular word)
# <li> Save 
# <li> Load
# <li> Sparcisity

# ### Load DataSet_vip_single_industry

# In[3]:


with open("D:3.AutoencoderForArticle/DataSet_vip_single_industry","rb") as f:
    DataSet=pickle.load(f)


# In[5]:


DataSet.info()


# In[9]:


DataSet.head()


# In[6]:


_=DataSet["title_token_plus_body_token"]
_=_.value_counts()
_.reset_index().rename(columns={"index":"title_token_plus_body_token","title_token_plus_body_token":"counts"})


# In[310]:


# with open("D:3.AutoencoderForArticle/testset_vip.v01.p","wb") as f:
#     pickle.dump(file=f,obj=trainset_vip)


# In[311]:


# with open("D:3.AutoencoderForArticle/testset_vip.v01.p","rb") as f:
#     x=pickle.load(f)


# ### Vectorization

# In[21]:


from keras.preprocessing.text import Tokenizer


# In[62]:


texts = DataSet["title_token_plus_body_token"]
tokenizer = Tokenizer(num_words=20000,)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# word_index


# In[29]:


_=pd.Series(tokenizer.word_counts)
_.sort_values(ascending=False,inplace=True)


# In[61]:


_[:20000][[0,1,2,3,4,5,-6,-5,-4,-3,-2,-1]]


# In[68]:


_=pd.Series(tokenizer.word_docs)
_.sort_values(ascending=False,inplace=True)
_[:20000][[0,1,2,3,4,5,-6,-5,-4,-3,-2,-1]]


# In[70]:


one_hot_results = tokenizer.texts_to_matrix(texts, mode='binary')
one_hot_results


# In[88]:


one_hot_results=one_hot_results.astype(int)


# In[90]:


one_hot_results


# ### Save one_hot_results 

# In[91]:


# np.save("D:3.AutoencoderForArticle/BOW_binary_v01.npy",one_hot_results)


# ### Load one_hot_results 

# In[6]:


BOW_binary=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[7]:


#sparcity
sparcity=np.mean(BOW_binary)
print("{}%".format(np.round(sparcity,5)*100))


# In[8]:


np.sum(BOW_binary[69520])


# ## Text to Vector version2
# <li> Load DataSet_vip_single_industry
# <li> Build Vocabulary(commenest 3700 words per industry)
# <li> Use Vocanbulry to vectrize text  
# <li> Sparcisity
# <li> Save

# #### Load DataSet_vip_single_industry

# In[ ]:


with open("D:3.AutoencoderForArticle/DataSet_vip_single_industry","rb") as f:
    DataSet=pickle.load(f)


# In[21]:


from keras.preprocessing.text import Tokenizer


# #### Build Vocabulary

# In[190]:


dictindustryvocabulary=dict()
n=3700
for idu in  tqdm_notebook(list_industry):
    text=DataSet[DataSet["indusrty_tags"]=={idu}]["title_token_plus_body_token"].values
    tokenizer=Tokenizer(num_words=n)
    tokenizer.fit_on_texts(text)
    
    df=pd.Series(tokenizer.word_index)
    wordset=df.sort_values(ascending=True).index[:n]
    
    dictindustryvocabulary[idu]=wordset


# In[291]:


vocabulary=set()
allhave=set()
for i,k in enumerate(dictindustryvocabulary):
#     print(len(dictindustryvocabulary[k]))
    vocabulary=vocabulary|set(dictindustryvocabulary[k])
    if i==0:allhave=allhave|set(dictindustryvocabulary[k])
    allhave=allhave&set(dictindustryvocabulary[k])


# In[292]:


len(vocabulary)


# In[310]:


word_industrys={i:0 for i in vocabulary}
for k in dictindustryvocabulary:
    for word in dictindustryvocabulary[k]:
        word_industrys[word]+=1


# In[311]:


df=pd.Series(word_industrys)
df=pd.DataFrame(df.sort_values(ascending=False))


# In[312]:


discard_list=[]
for word in vocabulary:
    if len(word)==1:discard_list.append(word)


# In[313]:


vocabulary=vocabulary-(set(discard_list)-set(element))
len(vocabulary)


# In[254]:


# with open("D:3.AutoencoderForArticle/vocabulary.p","wb") as f:
#     pickle.dump(file=f,obj=vocabulary)


# #### Use vocabulary vectorize text

# In[314]:


dict_vocabulary={w:i for i,w in enumerate(vocabulary)}


# In[315]:


text=DataSet["title_token_plus_body_token"]
BOW_binary=np.zeros((len(text),len(vocabulary)),dtype="int8")


# In[316]:


BOW_binary.shape


# In[318]:


for i,t in enumerate(tqdm_notebook(text)):
    for w in t.split():
        if w in dict_vocabulary:
            BOW_binary[i,dict_vocabulary[w]]=1


# #### Saprsity

# In[320]:


sparsity=np.sum(BOW_binary==1)/(BOW_binary.shape[0]*BOW_binary.shape[1])
print("sparsity:{:.2}%%".format(round(sparsity,5)*100))


# #### Save

# In[324]:


# np.save("D:3.AutoencoderForArticle/BOW_binary_v02.npy",BOW_binary)

