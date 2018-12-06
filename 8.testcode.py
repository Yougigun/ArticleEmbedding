
# coding: utf-8

# In[ ]:


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

from matplotlib import pyplot as plt

list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]
element="氫氦鋰鈹硼碳氮氧氟氖鈉鎂鋁矽磷硫氯氬鉀鈣鈧鈦釩鉻錳鐵鈷鎳銅鋅鎵鍺砷硒溴氪銣鍶銀鎘銦錫銻碲碘氙銫鋇鉑金汞鉈鉛鉍釙氡鍅鐳"


# In[ ]:


df=pd.read_hdf("Model/history.h5")


# In[ ]:


df[["loss","val_loss"]][:200].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[ ]:


df[["triplet_loss","val_triplet_loss"]][:200].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[ ]:


df[["anchor_loss","positive_loss","negative_loss"]][:50].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[ ]:


df[["val_anchor_loss","val_positive_loss","val_negative_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[ ]:


df[["loss","val_loss"]][:30].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[ ]:


df["triplet_loss"][342:343]


# In[ ]:


l=800
triplet_loss=pd.read_csv("run_model1-tag-triplet_loss.csv")
triplet_loss=triplet_loss[:l]["Value"]
val_triplet_loss=pd.read_csv("run_model1-tag-val_triplet_loss.csv")
val_triplet_loss=val_triplet_loss[:l]["Value"]

loss=pd.read_csv("run_model1-tag-loss.csv")
loss=loss[:l]["Value"]
val_loss=pd.read_csv("run_model1-tag-val_loss.csv")
val_loss=val_loss[:l]["Value"]

negative_loss=pd.read_csv("run_model1-tag-negative_loss.csv")
negative_loss=negative_loss[:l]["Value"]
val_negative_loss=pd.read_csv("run_model1-tag-val_negative_loss.csv")
val_negative_loss=val_negative_loss[:l]["Value"]

df2=pd.DataFrame({"loss":loss,"val_loss":val_loss,
                  "negative_loss":negative_loss,"val_negative_loss":val_negative_loss,
                  "triplet_loss":triplet_loss,"val_triplet_loss":val_triplet_loss})


# In[ ]:


_ = pd.DataFrame({20000:df["loss"][:l],19404:df2["loss"][:l]})
_.plot()


# In[ ]:


column="negative_loss"
_ = pd.DataFrame({20000:df[column][:l],19404:df2[column][:l]})
_.plot()


# In[ ]:


column="val_loss"
_ = pd.DataFrame({20000:df[column][:l],19404:df2[column][:l]})
_.plot()


# In[ ]:


column="val_triplet_loss"
_ = pd.DataFrame({20000:df[column][:l],19404:df2[column][:l]})
_.plot()


# ## Plot Tsene every 50 epochs

# #### load encoder model

# In[ ]:


from keras.models import load_model


# In[ ]:


def losspassfunction(y_true,y_pred):
    return y_pred
path="Models/Model2_addlayer/"
Tri_AutoEncoder=load_model(path+"Tri_AutoEncoder.initial.h5",custom_objects={"losspassfunction":losspassfunction})
Tri_AutoEncoder.load_weights(path+"regular/weights.1400.hdf5")
encoder=Tri_AutoEncoder.layers[3]


# #### load data

# In[ ]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v02.npy")


# In[ ]:


with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","rb") as f:
    train_dict_collect_small_industry=pickle.load(f)


# In[ ]:


with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","rb") as f:
    test_dict_collect_small_industry=pickle.load(f)


# In[ ]:


test_x=np.load("D:3.AutoencoderForArticle/test_x_v2.npy")
test_y=np.load("D:3.AutoencoderForArticle/test_y_v2.npy")


# #### Embedding

# In[ ]:


emnedding_test_x=encoder.predict(test_x)


# #### TSEN

# In[ ]:


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

tsne = TSNE(n_components=2, random_state=0,perplexity=30,n_iter=2000,
            metric=cosine,
            verbose=2)
intermediates_tsne=tsne.fit_transform(tsne_data)


# In[ ]:


#plot
color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]

plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
plt.legend(fontsize=12)
plt.savefig("tsnefig/fig1.png")


# In[ ]:


import os
def dot(u,v):
    u=np.asarray(u)
    v=np.asarray(v)
    uv=np.dot(u,v)
    sig=1/(1+np.exp(-uv))
    return 1-sig
def losspassfunction(y_true,y_pred):
    return y_pred
path="Models/Model2_addlayer/"

Tri_AutoEncoder=load_model(path+"Tri_AutoEncoder.initial.h5",custom_objects={"losspassfunction":losspassfunction})
encoder=Tri_AutoEncoder.layers[3]


metric=cosine
perplexity=30
n_iter=250
weightlist=os.listdir(path+"regular/")
tsen_result=[]
for j,w in enumerate(tqdm_notebook(weightlist)):
    Tri_AutoEncoder.load_weights(path+"regular/{}".format(w))
    encoder=Tri_AutoEncoder.layers[3]
    emnedding_test_x=encoder.predict(test_x)
    
    tsne_data=emnedding_test_x
    c=test_y
    
    tsne = TSNE(n_components=2, random_state=0,perplexity=perplexity,n_iter=n_iter,metric=metric,verbose=0)
    intermediates_tsne=tsne.fit_transform(tsne_data)
    tsen_result.append(intermediates_tsne)

    #plot
    color=["r","b","y","g","k"]
    marker=["^","s","o","D","+"]

    plt.figure(figsize=(8, 8),)
    for i,k in enumerate(set(c)):
        c=np.asarray(c)
        pick=c==k
        plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
    if j==0:plt.legend(fontsize=12)
    plt.xlim((-30,30))
    plt.ylim((-30,30))
    tsnpath =path+"tsnefigs/"
    if not os.path.isdir(tsnpath):
        os.mkdir(tsnpath)
    plt.savefig(tsnpath+"fig{}.png".format(j))
    


# In[ ]:


#plot
color=["r","b","y","g","k"]
marker=["^","s","o","D","+"]

plt.figure(figsize=(8, 8),)
for i,k in enumerate(set(c)):
    c=np.asarray(c)
    pick=c==k
    plt.scatter(x = intermediates_tsne[pick,0], y=intermediates_tsne[pick,1],c=color[i],marker=marker[i],label=k,)
# plt.legend(fontsize=12)
plt.xlim((-30,30))
plt.ylim((-30,30))
# plt.savefig("tsnefig/fig{}.png".format(j))

