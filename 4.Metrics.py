
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

from pprint import pprint
#coding:utf-8
import matplotlib.pyplot as plt

#有中文出现的情况，需要u'内容'
list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]


# In[3]:


from keras.layers import Input
from keras.models import Model


# ## Data Set

# In[5]:


picksample=np.load("D:3.AutoencoderForArticle/picksample.npy")
len(picksample)


# In[4]:


bow=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[5]:


with open("D:3.AutoencoderForArticle/dict_collect_industry.p","rb") as f:
    dict_collect_industry=pickle.load(f)


# ## Reconstruct Model

# In[6]:


from keras.models import load_model


# In[7]:


#Autoencoder
autoencoder_untrained=load_model("autoencoder.initial.h5")
autoencoder_trained=load_model("best_weights.hdf5")


# In[8]:


#Encoder_untrained
##layer
layer_encode=autoencoder_untrained.layers[1]
##dataflow
x=Input((20000,))
y=layer_encode(x)
##build Model
Encoder_untrained = Model(inputs=x,outputs=y)


# In[9]:


#Encoder_rained
##layer
layer_encode=autoencoder_trained.layers[1]
##dataflow
x=Input((20000,))
y=layer_encode(x)
##build Model
Encoder_trained = Model(inputs=x,outputs=y)


# ## mean parewise Similarity in same industry

# In[10]:


##除了同樣industry 也要比不同 industry 之間的


# In[11]:


dict_=dict()
for k in dict_collect_industry:
    dict_[k]=len(dict_collect_industry[k])
df=pd.Series(dict_)
x=df.sort_values(ascending=False)
x


# In[ ]:


import 


# ### Similarity  In terms of Bow

# In[12]:


dict_=dict()
for k in tqdm_notebook(dict_collect_industry):
    array=bow[dict_collect_industry[k]]
    if array.shape[0]!=0:
        dict_[k]=MeanSimilarityoneindustry(array)
y=pd.Series(dict_).sort_values(ascending=False)


# In[13]:


pd.DataFrame(data=[y,x]).T.rename(columns={0:"similarity",1:"counts"})


# ### Similarity  In terms of untrained AE

# In[26]:


dict_=dict()
for k in tqdm_notebook(dict_collect_industry):
    array=bow[dict_collect_industry[k]]
    if array.shape[0]!=0:
        array=Encoder_untrained.predict(array)
        dict_[k]=MeanSimilarityoneindustry(array)
y=pd.Series(dict_).sort_values(ascending=False)


# In[28]:


pd.DataFrame(data=[y,x]).T.rename(columns={0:"similarity",1:"counts"})


# ### Similarity  In terms of  AE

# In[29]:


dict_=dict()
for k in tqdm_notebook(dict_collect_industry):
    array=bow[dict_collect_industry[k]]
    if array.shape[0]!=0:
        array=Encoder_trained.predict(array)
        dict_[k]=MeanSimilarityoneindustry(array)
y=pd.Series(dict_).sort_values(ascending=False)


# In[30]:


pd.DataFrame(data=[y,x]).T.rename(columns={0:"similarity",1:"counts"})


# In[24]:


Encoder_untrained.predict(np.arange(20000*2).reshape((2,20000))).shape


# ## mean parewise Similarity in different industry

# ### Similarity  In terms of  Bow

# In[58]:


array1=bow[dict_collect_industry["食品飲料"]]
array2=bow[dict_collect_industry["造紙"]]
MeanSimilaritytwoindustry(array1,array2)


# In[59]:


li=[]
for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
#         array1=Encoder_trained.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
#                     array2=Encoder_trained.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[60]:


df=pd.DataFrame(li)
df=df.pivot(index=0,columns=1,values=2)


# In[61]:


df


# ### Similarity  In terms of untrained AE

# In[62]:


array1=Encoder_untrained.predict(bow[dict_collect_industry["食品飲料"]])
array2=Encoder_untrained.predict(bow[dict_collect_industry["造紙"]])
MeanSimilaritytwoindustry(array1,array2)


# In[63]:


li=[]
for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
        array1=Encoder_untrained.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
                    array2=Encoder_untrained.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[69]:


df_untrainAE=pd.DataFrame(li)
df_untrainAE=df_untrainAE.pivot(index=0,columns=1,values=2)


# In[70]:


df_untrainAE


# ### Similarity  In terms of  AE

# In[71]:


array1=Encoder_trained.predict(bow[dict_collect_industry["食品飲料"]])
array2=Encoder_trained.predict(bow[dict_collect_industry["造紙"]])
MeanSimilaritytwoindustry(array1,array2)


# In[72]:


li=[]
for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
        array1=Encoder_trained.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
                    array2=Encoder_trained.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[75]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[76]:


df_trainAE


# In[ ]:


# Metrics mean sum similarity one to multi


# In[130]:


#bow
li=[]
for i,S in df[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# In[132]:


#UAE
li=[]
for i,S in df_untrainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# In[133]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# In[136]:





# ### Similarity  In terms of  AE with supervision (Model one layer ,20000 common word)

# In[5]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[6]:


with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","rb") as f:
    train_dict_collect_small_industry=pickle.load(f)


# In[7]:


with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","rb") as f:
    test_dict_collect_small_industry=pickle.load(f)


# In[8]:


from keras.models import load_model


# In[9]:


# custommed function
def noisefunction(x):
    x_noise=K.ones_like(x)
    return K.in_train_phase(x_noise,x,1)

def tripletlossfunction(inputs):
    r01=inputs[0]
    r02=inputs[1]
    loss=K.log(1+K.exp(r02-r01))
#     x=np.array([[0],[1],[0],[1]])
#     x=K.variable(x)
    return loss

def losspassfunction(y_true,y_pred):
    return y_pred

def test(inputs):
#     x=K.dot(inputs,k.transpose(inputs))
    x=K.transpose(inputs)
    return x


# In[11]:


Tri_AutoEncoder=load_model("Tri_AutoEncoder_trained.h5",custom_objects={"losspassfunction":losspassfunction})


# In[12]:


encoder_in_AE=Tri_AutoEncoder.layers[3]


# ### on Trainset

# In[14]:


li=[]
model=encoder_in_AE
dict_collect_industry=train_dict_collect_small_industry

for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
        array1=model.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
                    array2=model.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[16]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[17]:


df_trainAE


# In[18]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# #### on testset

# In[21]:


li=[]
model=encoder_in_AE
dict_collect_industry=test_dict_collect_small_industry

for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
        array1=model.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
                    array2=model.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[22]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[23]:


df_trainAE


# In[25]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# ### Similarity  In terms of  AE with supervision (Model 2 layer ,common 19404 words)

# In[2]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v02.npy")


# In[7]:


with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","rb") as f:
    train_dict_collect_small_industry=pickle.load(f)


# In[8]:


with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","rb") as f:
    test_dict_collect_small_industry=pickle.load(f)


# In[9]:


from keras.models import load_model


# In[10]:


# custommed function
def noisefunction(x):
    x_noise=K.ones_like(x)
    return K.in_train_phase(x_noise,x,1)

def tripletlossfunction(inputs):
    r01=inputs[0]
    r02=inputs[1]
    loss=K.log(1+K.exp(r02-r01))
#     x=np.array([[0],[1],[0],[1]])
#     x=K.variable(x)
    return loss

def losspassfunction(y_true,y_pred):
    return y_pred

def test(inputs):
#     x=K.dot(inputs,k.transpose(inputs))
    x=K.transpose(inputs)
    return x


# In[11]:


Tri_AutoEncoder=load_model("Models/Model2_addlayer/bestmodel.hdf5",custom_objects={"losspassfunction":losspassfunction})


# In[12]:


encoder_in_AE=Tri_AutoEncoder.layers[3]


# ### on Trainset

# In[17]:


li=[]
model=encoder_in_AE
dict_collect_industry=train_dict_collect_small_industry
bow=Data
for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
        array1=model.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
                    array2=model.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[18]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[19]:


df_trainAE


# In[18]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# ### on testset

# In[13]:


li=[]
model=encoder_in_AE
dict_collect_industry=test_dict_collect_small_industry
bow=Data
for k in tqdm_notebook(dict_collect_industry):
    array1=bow[dict_collect_industry[k]]
    if len(array1)!=0:
        array1=model.predict(array1)
        for j in dict_collect_industry:
                array2=bow[dict_collect_industry[j]]
                if len(array2)!=0:
                    array2=model.predict(array2)
                    simi=MeanSimilaritytwoindustry(array1,array2)
                    print(k,j,simi)
                    li.append((k,j,simi))


# In[14]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[15]:


df_trainAE


# In[16]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)

