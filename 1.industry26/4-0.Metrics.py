
# coding: utf-8

# In[161]:


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
import seaborn as sns

#solved chinese display in matplotlib
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' 
mpl.rcParams['axes.unicode_minus'] = False


list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]


# In[292]:


def IQR(array):
    Q3,Q1=np.percentile(array,[75,25])
    return Q3-Q1


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
def tripletlossfunction(inputs):
    r01=inputs[0]
    r02=inputs[1]
    loss=K.log(1+K.exp(r02-r01))
#     x=np.array([[0],[1],[0],[1]])
#     x=K.variable(x)
    return loss

def losspassfunction(y_true,y_pred):
    return y_pred


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

# In[29]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v02.npy")


# In[80]:


with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","rb") as f:
    train_dict_collect_small_industry=pickle.load(f)


# In[81]:


with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","rb") as f:
    test_dict_collect_small_industry=pickle.load(f)


# #### load model

# In[82]:


from keras.models import load_model


# In[83]:


# custommed function
def tripletlossfunction(inputs):
    r01=inputs[0]
    r02=inputs[1]
    loss=K.log(1+K.exp(r02-r01))
#     x=np.array([[0],[1],[0],[1]])
#     x=K.variable(x)
    return loss

def losspassfunction(y_true,y_pred):
    return y_pred


# In[84]:


Tri_AutoEncoder=load_model("Models/Model2_addlayer/Tri_AutoEncoder_trained.h5",custom_objects={"losspassfunction":losspassfunction})


# In[85]:


encoder_in_AE=Tri_AutoEncoder.layers[3]


# ### on Testset

# In[86]:


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


# In[87]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[88]:


df_trainAE


# In[89]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# ### on Trainset

# In[90]:


li=[]
model=encoder_in_AE
dict_collect_industry=train_dict_collectindustry
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


# In[91]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[92]:


df_trainAE


# In[93]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# ## Similarity  In terms of  AE with supervision (1 layer ,common 19404 words,50000 news)

# In[96]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v02.npy")


# In[97]:


with open("D:3.AutoencoderForArticle/train_dict_collect_industry_50000.p","rb") as f:
    train_dict_collect_industry=pickle.load(f)


# In[98]:


with open("D:3.AutoencoderForArticle/test_dict_collect_industry_50000.p","rb") as f:
    test_dict_collect_industry=pickle.load(f)


# In[99]:


from keras.models import load_model


# In[100]:


def losspassfunction(y_true,y_pred):
    return y_pred


# In[101]:


Tri_AutoEncoder=load_model("Models/Model3_on_all_industry/bestmodel.hdf5",custom_objects={"losspassfunction":losspassfunction})


# In[102]:


encoder_in_AE=Tri_AutoEncoder.layers[3]


# ### on Trainset

# In[72]:


li=[]
model=encoder_in_AE
dict_collect_industry=train_dict_collect_industry
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


# In[73]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[75]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# ### on testset

# In[103]:



li=[]
model=encoder_in_AE
dict_collect_industry=test_dict_collect_industry
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


# In[104]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[105]:


df_trainAE


# In[106]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# ## Standard deviation and mean of Simiarity In terms of  AE with supervision (1 layer ,common 19404 words,50000 news)

# In[3]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v02.npy")


# In[4]:


with open("D:3.AutoencoderForArticle/train_dict_collect_industry_50000.p","rb") as f:
    train_dict_collect_industry=pickle.load(f)


# In[5]:


with open("D:3.AutoencoderForArticle/test_dict_collect_industry_50000.p","rb") as f:
    test_dict_collect_industry=pickle.load(f)


# In[6]:


from keras.models import load_model


# In[7]:


def losspassfunction(y_true,y_pred):
    return y_pred


# In[8]:


Tri_AutoEncoder=load_model("Models/Model3_on_all_industry/bestmodel.hdf5",custom_objects={"losspassfunction":losspassfunction})


# In[9]:


encoder_in_AE=Tri_AutoEncoder.layers[3]


# #### Mean Similarity on train_dict_collect_industry

# In[10]:


li=[]
model=encoder_in_AE
dict_collect_industry=train_dict_collect_industry
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


# In[11]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[12]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)


# #### Violin & Boxplot on trainset

# In[96]:


index=train_dict_collect_industry["科技相關"]
train_x=Data[index]


# In[97]:


em_train_x=model.predict(train_x)


# In[98]:


em_train_x.shape


# In[99]:


simi=cosine_similarity(em_train_x)


# In[272]:


serieslist=[]
labels=[]
for k in tqdm_notebook(list(train_dict_collect_industry.keys())):
    index=train_dict_collect_industry[k]
    train_x=Data[index]
    em_train_x=model.predict(train_x)
    simi=cosine_similarity(em_train_x)
    series=simi[np.tri(simi.shape[0],simi.shape[1],k=-1)==1] #k should be -1
    serieslist.append(series)
    labels.append(k)


# In[143]:


fig=plt.figure(figsize=(12,6))
# plt.xticks([2,3,4])
plt.boxplot(serieslist,showmeans=True,vert=True,labels=labels)


# In[155]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=serieslist[:12],orient="v")
ax.set_xticklabels(labels[:12])


# In[164]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=serieslist[12:],orient="v")
ax.set_xticklabels(labels[12:])


# In[174]:


index=labels.index("科技相關")
fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=serieslist[index:index+1],orient="v")
ax.set_xticklabels(labels[index:index+1])


# #### similarity disrtrbution on testset

# In[176]:


li=[]
model=encoder_in_AE
dict_collect_industry=test_dict_collect_industry
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


# In[284]:


serieslist=[]
labels=[]

for k in tqdm_notebook(list(test_dict_collect_industry.keys())):
    index=test_dict_collect_industry[k]
    test_x=Data[index]
    em_test_x=model.predict(test_x)
    simi=cosine_similarity(em_test_x)
    series=simi[np.tri(simi.shape[0],simi.shape[1],k=-1)==1]
#     series=simi
    serieslist.append(series)
    labels.append(k)


# In[178]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=serieslist[:12],orient="v")
ax.set_xticklabels(labels[:12])


# In[179]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=serieslist[12:],orient="v")
ax.set_xticklabels(labels[12:])


# #### dissimilarity distribution

# In[188]:


index1=test_dict_collect_industry["化學工業"]
index2=test_dict_collect_industry["公用事業"]
x1=Data[index1]
x2=Data[index2]
em_x1=model.predict(x1)
em_x2=model.predict(x2)
simi=cosine_similarity(em_x1,em_x2)
simi=simi.reshape(1,-1)


# In[189]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=simi,orient="v")


# In[283]:


dissimilarity=[]
industry=[]
dict_collect_industry=test_dict_collect_industry
for k in tqdm_notebook(list(dict_collect_industry.keys())):
    index1=dict_collect_industry[k]
    x1=Data[index1]
    em_x1=model.predict(x1)
    simitotla=[]
    for j in list(dict_collect_industry.keys()):
        if k!=j:
            index2=dict_collect_industry[j]
            x2=Data[index2]
            em_x2=model.predict(x2)
            simi=cosine_similarity(em_x1,em_x2)
            simi=list(simi.flatten())
            simitotla+=simi
    dissimilarity.append(simitotla)  
    industry.append(k)


# In[219]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=dissimilarity[:12],orient="v",)
ax.set_xticklabels(industry[:12])


# In[220]:


fig=plt.figure(figsize=(18,6))
ax=sns.violinplot(data=dissimilarity[12:],orient="v")
ax.set_xticklabels(industry[12:])


# ## R,UR,(U-UR)/2,Simi-Quortail,Dissimi-Quortail on testset

# In[221]:


li=[]
model=encoder_in_AE
dict_collect_industry=test_dict_collect_industry
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


# In[222]:


df_trainAE=pd.DataFrame(li)
df_trainAE=df_trainAE.pivot(index=0,columns=1,values=2)


# In[365]:


#AE
li=[]
for i,S in df_trainAE[:].iterrows():
    R=S.loc[i]
    UR=S.drop(i).mean()
    result=(R-UR)/2
    li.append((i,R,UR,result))
df=pd.DataFrame(li).rename(columns={0:"industry",1:"R",2:"UR",3:"(R-UR)/2"}).sort_values("(R-UR)/2",ascending=False)
# df=df.drop("控股")
df=df.set_index("industry")
df=df.drop("控股")
df


# #### add IQR,STD,Q3,Q1

# In[326]:


SimilarityDistribution={k:np.asarray(d) for k,d in zip(labels,serieslist)}
DissimilarityDistribution={k:np.asarray(d) for k,d in zip(industry,dissimilarity)}
SimilarityDistribution.pop("控股","nothiskey")
DissimilarityDistribution.pop("控股","nothiskey")


# In[371]:


simiquartile={k : np.round(np.percentile(SimilarityDistribution[k],[75,50,25]),3) for k in SimilarityDistribution }
dissimiquartile={k : np.round(np.percentile(DissimilarityDistribution[k],[75,50,25]),3) for k in DissimilarityDistribution }


# In[364]:


SimiQ=pd.DataFrame(simiquartile,index=["Q3","Q2","Q1"]).T
SimiQ


# In[370]:


df["Simi-Q3"]=SimiQ["Q3"]
df["Simi-Q2"]=SimiQ["Q2"]
df["Simi-Q1"]=SimiQ["Q1"]
df["Simi-IQR"]=df["Simi-Q3"]-df["Simi-Q1"]
df


# In[372]:


DissimiQ=pd.DataFrame(dissimiquartile,index=["Q3","Q2","Q1"]).T
DissimiQ


# In[373]:


df["Dissimi-Q3"]=DissimiQ["Q3"]
df["Dissimi-Q2"]=DissimiQ["Q2"]
df["Dissimi-Q1"]=DissimiQ["Q1"]
df["Dissimi-IQR"]=df["Dissimi-Q3"]-df["Dissimi-Q1"]
df[]


# In[386]:


df.describe()[["R","UR","(R-UR)/2","Simi-Q1","Dissimi-Q3"]].loc[["mean"]].T

