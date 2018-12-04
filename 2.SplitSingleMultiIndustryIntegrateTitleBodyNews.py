
# coding: utf-8

# In[162]:


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

import re

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


# ### 新聞數牽扯到所有產業

# In[123]:


dict_={k:0 for k in list_industry }
for i in tqdm_notebook(News_vip_with_at_least_one_industrytag["indusrty_tags"]):
#     print(i)
    for k in dict_ :
        if k in i : dict_[k]=dict_[k]+1
ser=pd.Series(dict_)
ser=ser.sort_values(ascending=False)
ser


# In[124]:


##總和
ser.sum()


# ### 新聞數牽扯到單一產業

# In[120]:


#每則新聞分類
dict_=dict()
for t in News_vip_with_at_least_one_industrytag["indusrty_tags"]:
    t=",".join(list(t))
    if t not in dict_:dict_[t]=1
    else: dict_[t]+=1
#只找出單一產業新聞
dict__={k:0 for k in list_industry }
for i in pd.Series(dict_).index:
    if len(i.split(","))==1:dict__[i]=pd.Series(dict_).loc[i]

ser=pd.Series(dict__)
ser.sort_values(ascending=False)


# In[121]:


##總和
ser.sum()


# # Build DataSet for Single industry News

# In[158]:


# Pick up the news only involves one news and retain ["guid","title_token","body_token","indusrty_tags"] columns
_ = News_vip_with_at_least_one_industrytag["indusrty_tags"]
pick=_.progress_apply(lambda x : True if len(x)==1 else False)

DataSet=News_vip_with_at_least_one_industrytag[pick][["guid","title_token","body_token","indusrty_tags"]]
DataSet.reset_index(drop=True,inplace=True) #Reset index form 0 to ~
DataSet


# In[177]:


# Merge Title and body token
DataSet["title_token_plus_body_token"]=DataSet["title_token"]+" "+DataSet["body_token"]
DataSet["title_token_plus_body_token"][12500]


# In[178]:


#clean token
def CleanToken(string):
    pattern=[re.compile("[\)》\，\.\、\-\%\《\(\"\'\％\」\「\。\（\）\；\● ]"),
         re.compile("\d\d%"),##EX: 85%
         re.compile("\d+"),##EX:9,10,123..
         re.compile(" [a-zA-Z] "),##EX: x , d ,...
#          re.compile("[a-zA-Z]"),
        ]
# string=trainset_vip["Title_and_body"][7000]
    for p in pattern:
        string=p.sub(" ",string)
    string=re.sub("  +"," ",string)# two or above space to one space
    string=re.sub("^ ","",string)#space at beginning
    string=re.sub(" $","",string)#space at end
    return string
DataSet["title_token_plus_body_token"]=DataSet["title_token_plus_body_token"].progress_apply(CleanToken)


# In[181]:


# pick some columns ["guid","title_token_plus_body_token","indusrty_tags"]
DataSet=DataSet[["guid","title_token_plus_body_token","indusrty_tags"]]


# ## Save DataSet

# In[182]:


# # DataSet.reset_index(drop=True,inplace=True)
# with open("D:3.AutoencoderForArticle/DataSet_vip_single_industry","wb") as f:
#     pickle.dump(file=f,obj=DataSet)

