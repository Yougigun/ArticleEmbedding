
# coding: utf-8

# In[1]:


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


# In[342]:


small_triplet={k:[] for k in small_list_insustry}
for k in small_triplet:
    positive=small_dict_collect_industry[k]
    negative=negative_pool[k]
    for i in range(len(positive[:-1])):
        small_triplet[k].append(positive[i:i+2]+list(negative[i:i+1]))
    assert len(small_triplet[k])==len(small_dict_collect_industry[k])-1
dict_small_triplet=small_triplet


# ####   anchor positive negative
# ####   (A1,A2,C9)
# ####   (A2,A3,C5)
# ####   ....etc.
# ####   note:randomly pick negative
# ####   disadvantage: negative too less , may cause overfitting or even cant generalization

# In[343]:


# with open("D:3.AutoencoderForArticle/dict_small_triplet.p","wb") as f:
#     pickle.dump(file=f,obj=dict_small_triplet)


# In[344]:


# with open("D:3.AutoencoderForArticle/dict_small_triplet.p","rb") as f:
#     test=pickle.load(f)


# In[345]:





# ## V2 tripletDataset

# In[2]:


with open("D:3.AutoencoderForArticle/dict_collect_industry.p","rb") as f:
    dict_collect_industry=pickle.load(f)


# In[27]:


small_list_insustry=["車輛相關","生技醫療保健","營建地產","百貨通路","傳播出版"]
dict_collect_small_industry={i:dict_collect_industry[i] for i in small_list_insustry }
train_dict_collect_small_industry=dict()
test_dict_collect_small_industry=dict()
for k in dict_collect_small_industry:
    split=0.8
    indulist=dict_collect_small_industry[k]
    indulen=len(indulist)
    train_dict_collect_small_industry[k]=indulist[:int(indulen*split)]
    test_dict_collect_small_industry[k]=indulist[int(indulen*split):]


# In[28]:


get_ipython().run_cell_magic('time', '', 'train_dict_small_triplet_v02=dict()\nfor k in tqdm_notebook(train_dict_collect_small_industry):\n    poslist=train_dict_collect_small_industry[k]\n    poslen=len(poslist)\n    neglist=[]\n    for j in train_dict_collect_small_industry:\n        if k!=j :\n            neglist+=train_dict_collect_small_industry[j]\n        neglen=len(neglist)\n    indarray=np.zeros((int(poslen*(poslen-1)*neglen/2),3),dtype=int)\n    \n    i=0\n    for ip1_,p1 in enumerate(tqdm_notebook(poslist)):\n        for ip2_,p2 in enumerate(poslist[ip1_+1:]):\n            for in_,n in enumerate(neglist):\n#                 print(p1,p2,n)\n                indarray[i,0]=p1\n                indarray[i,1]=p2\n                indarray[i,2]=n\n                i+=1\n#                 break\n#             break\n#         break\n    train_dict_small_triplet_v02[k]=indarray')


# In[30]:


get_ipython().run_cell_magic('time', '', 'test_dict_small_triplet_v02=dict()\nfor k in tqdm_notebook(test_dict_collect_small_industry):\n    poslist=test_dict_collect_small_industry[k]\n    poslen=len(poslist)\n    neglist=[]\n    for j in test_dict_collect_small_industry:\n        if k!=j :\n            neglist+=test_dict_collect_small_industry[j]\n        neglen=len(neglist)\n    indarray=np.zeros((int(poslen*(poslen-1)*neglen/2),3),dtype=int)\n    \n    i=0\n    for ip1_,p1 in enumerate(tqdm_notebook(poslist)):\n        for ip2_,p2 in enumerate(poslist[ip1_+1:]):\n            for in_,n in enumerate(neglist):\n#                 print(p1,p2,n)\n                indarray[i,0]=p1\n                indarray[i,1]=p2\n                indarray[i,2]=n\n                i+=1\n#                 break\n#             break\n#         break\n    test_dict_small_triplet_v02[k]=indarray')


# In[32]:


test_dict_small_triplet_v02[k].shape


# In[33]:


train_dict_small_triplet_v02[k].shape


# In[42]:


# with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","wb") as f:
#     pickle.dump(train_dict_collect_small_industry,f)


# In[43]:


# with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","wb") as f:
#     pickle.dump(test_dict_collect_small_industry,f)


# In[44]:


# with open("D:3.AutoencoderForArticle/train_dict_small_triplet_v02","wb") as f:
#     pickle.dump(train_dict_small_triplet_v02,f)


# In[45]:


# with open("D:3.AutoencoderForArticle/test_dict_small_triplet_v02","wb") as f:
#     pickle.dump(test_dict_small_triplet_v02,f)


# In[49]:


pprint(test)

