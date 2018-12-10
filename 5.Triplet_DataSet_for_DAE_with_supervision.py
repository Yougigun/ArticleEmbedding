
# coding: utf-8

# In[2]:


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
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'
list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]
small_list_insustry=["車輛相關","生技醫療保健","營建地產","百貨通路","傳播出版"]


# In[180]:


picksample=np.load("D:3.AutoencoderForArticle/picksample.npy")
len(picksample)


# In[181]:


bow=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[182]:


with open("D:3.AutoencoderForArticle/dict_collect_industry.p","rb") as f:
    dict_collect_industry=pickle.load(f)


# In[193]:


dict_={k:len(dict_collect_industry[k]) for k in dict_collect_industry}
pd.Series(dict_).sort_values(ascending=False)


# In[223]:


train_test_dict_collect_industry=dict()
for k in dict_collect_industry:
    split=int(len(dict_collect_industry[k])*0.8)
    train_test_dict_collect_industry["train_{}".format(k)]=dict_collect_industry[k][:split]
    train_test_dict_collect_industry["test_{}".format(k)]=dict_collect_industry[k][split:] 


# In[224]:


dict_={k:len(train_test_dict_collect_industry[k]) for k in train_test_dict_collect_industry}
pd.Series(dict_).sort_values(ascending=False)


# In[225]:


# #Save train_test_dict_collect_industry
# with open("D:3.AutoencoderForArticle/train_test_dict_collect_industry.p","wb") as f:
#     pickle.dump(file=f,obj=train_test_dict_collect_industry)


# In[244]:


small_list_insustry=["車輛相關","生技醫療保健","營建地產","百貨通路","傳播出版"]


# In[236]:


small_train_test_dict_collect_industry=dict()
for i in test_insustry:
    small_train_test_dict_collect_industry["train_{}".format(i)]=train_test_dict_collect_industry["train_{}".format(i)]
    small_train_test_dict_collect_industry["test_{}".format(i)]=train_test_dict_collect_industry["test_{}".format(i)]
    


# In[243]:


triplet_train_test_dict_collect_industry=dict()
for i in test_insustry:
    triplet_train_test_dict_collect_industry["train_{}".format(i)]=[]
    triplet_train_test_dict_collect_industry["test_{}".format(i)]=[]


# In[240]:


negative=dict()
for k in small_train_test_dict_collect_industry:
    print(k[:5])


# In[246]:


small_dict_collect_industry={k:dict_collect_industry[k] for k in small_list_insustry}


# In[294]:


negative_pool=dict({k:[] for k in small_dict_collect_industry })
for k in small_dict_collect_industry:
    for j in small_dict_collect_industry:
        if k!=j:negative_pool[k]+=small_dict_collect_industry[j]
    if len(negative_pool[k])!=0:negative_pool[k]=np.random.permutation(negative_pool[k])


# ## Build dict train/test industry:index in 50,000 single industry VIP news  

# In[8]:


with open("D:3.AutoencoderForArticle/DataSet_vip_single_industry","rb") as f:
    Dataset=pickle.load(f)


# In[10]:


Dataset.info()


# #### build dict_collect_industry_50000

# In[121]:


np.random.seed(0)
suffleindex=np.random.permutation(len(Dataset))
pick=suffleindex[:50000]
pick


# In[122]:


dict_collect_industry_50000={i:[] for i in list_industry }
for i,j in Dataset.loc[pick].iterrows():
    for k in dict_collect_industry_50000:
        if k in j[2]:
            dict_collect_industry_50000[k].append(i)
            break


# In[123]:


dict_count={}
for k in dict_collect_industry_50000:
    dict_count[k]=len(dict_collect_industry_50000[k])
df=pd.Series(dict_count)
df.sort_values(ascending=False)


# ### build train and test

# In[172]:


train_dict_collect_industry_50000=dict()
test_dict_collect_industry_50000=dict()
np.random.seed(10)
rate=3/4
for i in dict_collect_industry_50000:
    induslist=np.random.permutation(dict_collect_industry_50000[i])
    if 0!=len(induslist):
        split=int(np.ceil(len(induslist)*rate))
        train_dict_collect_industry_50000[i]=induslist[:split]
        test_dict_collect_industry_50000[i]=induslist[split:]


# #### Save train_test_dict_collect_industry

# In[176]:


with open("D:3.AutoencoderForArticle/train_dict_collect_industry_50000.p","wb") as f:
    pickle.dump(file=f,obj=train_dict_collect_industry_50000)
with open("D:3.AutoencoderForArticle/test_dict_collect_industry_50000.p","wb") as f:
    pickle.dump(file=f,obj=test_dict_collect_industry_50000)


# In[173]:


dict_count={}
dict_idu_list=train_dict_collect_industry_50000
for k in dict_idu_list:
    dict_count[k]=len(dict_idu_list[k])
df=pd.Series(dict_count)
df.sort_values(ascending=False)


# In[174]:


dict_count={}
dict_idu_list=test_dict_collect_industry_50000
for k in dict_idu_list:
    dict_count[k]=len(dict_idu_list[k])
df=pd.Series(dict_count)
df.sort_values(ascending=False)

