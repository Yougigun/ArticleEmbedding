
# coding: utf-8

# In[1]:


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


# In[2]:


list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]


# ### Load dict_industry

# In[14]:


with open("D:dict_industry","rb") as f:
    dict_industry=pickle.load(f)


# In[15]:


dict_industry["金融相關"]


# # Load Vip NEWS with industrytag

# In[16]:


with open("D:News_vip_with_industrytag","rb") as f:
    News_vip_with_industrytag=pickle.load(f)


# In[17]:


News_vip_with_industrytag.info()


# In[20]:


News_vip_with_industrytag[["publishtime","title","tags_cn","indusrty_tags"]][40:40]


# #### How many articles have at least one industry tags

# In[63]:


(News_vip_with_industrytag["indusrty_tags"]!=set()).value_counts()


# #### Drop Atricles without Industry tags

# In[68]:


pick=News_vip_with_industrytag["indusrty_tags"]!=set()
News_vip_with_at_least_one_industrytag=News_vip_with_industrytag[pick].reset_index()
News_vip_with_at_least_one_industrytag["indusrty_tags"]


# ##### How many articles have same body

# In[91]:


df=News_vip_with_at_least_one_industrytag["body"].value_counts().reset_index().rename(columns={"index":"body","body":"counts"})
df.head(10)


# In[93]:


News_vip_with_at_least_one_industrytag[News_vip_with_at_least_one_industrytag["body"]==df["body"].values[1]]


# ##### How many articles have same body_token

# In[94]:


df=News_vip_with_at_least_one_industrytag["body_token"].value_counts().reset_index().rename(columns={"index":"body_token","body_token":"counts"})
df.head(10)


# In[80]:


News_vip_with_at_least_one_industrytag[News_vip_with_at_least_one_industrytag["body_token"]==df["body_token"].values[1]]


# In[96]:


dict_=dict()
for t in News_vip_with_at_least_one_industrytag["indusrty_tags"]:
    t=",".join(list(t))
    if t not in dict_:dict_[t]=1
    else: dict_[t]+=1


# In[102]:


pd.Series(dict_).sort_values(ascending=False)


# ### 列出只有單一產業新聞

# In[100]:


dict__=dict()
for i in pd.Series(dict_).index:
    if len(i.split(","))==1:dict__[i]=pd.Series(dict_).loc[i]
##單看單一產業新聞
ser=pd.Series(dict__)
ser.sort_values(ascending=False)


# In[104]:


##總和
ser.sum()


# # Build Testset and Trainset

# In[65]:


seed=26
np.random.seed(seed)
shuffle=np.random.permutation(len(News_vip_with_industrytag))


# In[66]:


News_vip_with_industrytag_shuffle=News_vip_with_industrytag.iloc[shuffle]
News_vip_with_industrytag_shuffle=News_vip_with_industrytag_shuffle.iloc[shuffle]
News_vip_with_industrytag_shuffle


# In[73]:


TestSet=News_vip_with_industrytag_shuffle[:int(len(News_vip_with_industrytag_shuffle)*(2/10))]
TestSet


# In[69]:


dict_={i:0 for i in list_industry}
for i in TestSet["tags"]:
    for j in dict_:
        if j in i:
            dict_[j]+=1 
df=pd.Series(data=dict_)
df.sort_values(ascending=False)


# ## Save testset

# In[72]:


# TestSet.reset_index(drop=True,inplace=True)
# with open("D:3.AutoencoderForArticle/testset_vip","wb") as f:
#     pickle.dump(file=f,obj=TestSet)


# In[75]:


TrainSet=News_vip_with_industrytag_shuffle[int(len(News_vip_with_industrytag_shuffle)*(2/10)):]
TrainSet


# ## Save testset

# In[79]:


# TrainSet.reset_index(drop=True,inplace=True)
# with open("D:3.AutoencoderForArticle/trainset_vip","wb") as f:
#     pickle.dump(file=f,obj=TrainSet)

