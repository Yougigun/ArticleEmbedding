
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
# warnings.simplefilter(action='ignore', category=FutureWarning,)
# %matplotlib inline

import re

from pprint import pprint
#coding:utf-8
import matplotlib.pyplot as plt

list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]


# In[2]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[3]:


with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","rb") as f:
    train_dict_collect_small_industry=pickle.load(f)


# In[4]:


with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","rb") as f:
    test_dict_collect_small_industry=pickle.load(f)


# In[5]:


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


# In[12]:


encoder=load_model("encoder_trained.h5")
Tri_AutoEncoder=load_model("Tri_AutoEncoder_trained.h5",custom_objects={"losspassfunction":losspassfunction})


# In[119]:


set(y)


# In[165]:


train_x=Data[train_x_index]
train_embeddings=encoder_in_AE.predict(train_x)


# ## Tsne on Original Data

# In[264]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0,perplexity=40,verbose=1)
intermediates_tsne = tsne.fit_transform(train_x)


# In[265]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ## Tsne on embedding of Train data
# <li>Reconstruct Encoder
# <li>PaPrepare train data for Tsne
# <li>metric Euclidean metric
# <li>metric Euclidean metric
# <li>metric Euclidean metric

# ### Reconstruct Encoder

# In[175]:


encoder_in_AE=Tri_AutoEncoder.layers[3]


# ### Prepare data for Tsne

# In[161]:


train_x_index=[]
train_y=[]
for k in train_dict_collect_small_industry:
    newslist=train_dict_collect_small_industry[k]
    num=len(newslist)
    train_x_index+=newslist
    _=[k]*num
    train_y+=_


# ### metric distance

# In[262]:


from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
#set
tsne_embeddings=train_embeddings
c=train_y
#trian
tsne = TSNE(n_components=2, random_state=0,perplexity=40,verbose=1)
intermediates_tsne=tsne.fit_transform(tsne_embeddings)



# In[263]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ### metric cosine

# In[260]:


from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
#set
tsne_embeddings=train_embeddings
c=train_y
#trian
tsne = TSNE(n_components=2, random_state=0,perplexity=40,verbose=1,metric=cosine)
intermediates_tsne=tsne.fit_transform(tsne_embeddings)



# In[261]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ### metric dot

# In[258]:


from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

#metric
def dot(u,v):
    u=np.asarray(u)
    v=np.asarray(v)
    uv=np.dot(u,v)
    sig=1/(1+np.exp(-uv))
    return 1-sig
#set
tsne_embeddings=train_embeddings
c=train_y
#trian
tsne = TSNE(n_components=2, random_state=0,perplexity=40,verbose=1,metric=dot)
intermediates_tsne=tsne.fit_transform(tsne_embeddings)


# In[259]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ## Tsne on embedding of test data
# <li>prepare test set for tsen(must)
# <li>metric Euclidean metric
# <li>metric Euclidean metric
# <li>metric Euclidean metric

# ### prepare test set for tsen

# In[1]:


test_x_index=[]
test_y=[]
for k in test_dict_collect_small_industry:
    newslist=test_dict_collect_small_industry[k]
    num=len(newslist)
    test_x_index+=newslist
    _=[k]*num
    test_y+=_

test_x=Data[test_x_index]
test_embeddings=encoder_in_AE.predict(test_x)


# ### metric Euclidean metric 

# In[256]:


from sklearn.manifold import TSNE
#set
tsne_embeddings=test_embeddings
c=test_y

tsne = TSNE(n_components=2, random_state=0,perplexity=30,n_iter=1000,verbose=1)
intermediates_tsne = tsne.fit_transform(embeddings)


# In[257]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ### metric cosine metric 

# In[254]:


from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
#set
tsne_data=test_embeddings
c=test_y
#train
tsne = TSNE(n_components=2, random_state=0,perplexity=40,metric=cosine,verbos=1)
intermediates_tsne=tsne.fit_transform(tsne_data)


# In[255]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ### metric dot metric 

# In[252]:


from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
#set
tsne_data=test_embeddings
c=test_y
#metric
def dot(u,v):
    u=np.asarray(u)
    v=np.asarray(v)
    uv=np.dot(u,v)
    sig=1/(1+np.exp(-uv))
    return 1-sig

tsne = TSNE(n_components=2, random_state=0,perplexity=40,metric=dot,verbose=2)
intermediates_tsne=tsne.fit_transform(tsne_data)


# In[253]:


#plot
color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# In[251]:


color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]
from pylab import mpl
mpl.rcParams["font.family"] = 'DFKai-sb' # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

ax=plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)


# ## TSNE of TestSet(12491 from 50000)

# ### Load data

# In[4]:


train_x=np.load("D:3.AutoencoderForArticle/train_x_v2_50000.npy")
train_y=np.load("D:3.AutoencoderForArticle/train_y_v2_50000.npy")
test_x=np.load("D:3.AutoencoderForArticle/test_x_v2_50000.npy")
test_y=np.load("D:3.AutoencoderForArticle/test_y_v2_50000.npy")


# #### Embedding

# In[114]:


# np.random.seed(0)
pick=np.random.permutation(len(train_x))[:1000]
train_x_pick=train_x[pick]
train_y_pick=train_y[pick]


# In[215]:


np.random.seed(0)
pick=np.random.permutation(len(test_y))[:1000]
test_x_pick=test_x[pick]
test_y_pick=test_y[pick]
pick


# In[277]:


emnedding_test_x=encoder.predict(test_x)
emnedding_test_x.shape


# In[315]:


from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
#set
tsne_data=emnedding_test_x
c=test_y
#metric
def dot(u,v):
    u=np.asarray(u)
    v=np.asarray(v)
    uv=np.dot(u,v)
    sig=1/(1+np.exp(-uv))
    return 1-sig

tsne = TSNE(n_components=2, random_state=0,
            perplexity=50,
            n_iter=1000,
            metric=cosine,
            verbose=2)
intermediates_tsne=tsne.fit_transform(tsne_data)


# In[317]:


#plot
color=["#ff6f52","#3778bf","#ed0dd9","#feb209","#a90308","#758000"]
marker=["^","s","o","D","+","x"]
import matplotlib
plt.figure(figsize=(20,20),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],
                c=color[i%len(color)],s=40,linewidth=1,edgecolors="black",
#                 cmap="flag",
                marker=marker[(i//len(marker))%len(marker)],
                label=k,)
plt.legend(fontsize=19,
#            mode="expand",
           ncol=6,
           loc='lower left',
           bbox_to_anchor=(0,1),fancybox=True,shadow=True)
plt.xlim((-75,75))
plt.ylim((-75,75))
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig("tsnefig/cosine.test.png")

