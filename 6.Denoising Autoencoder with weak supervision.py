
# coding: utf-8

# In[72]:


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
warnings.simplefilter(action='ignore', category=FutureWarning,)
# %matplotlib inline

import re

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


# In[73]:


from keras.layers import Dense,Lambda,Input,Dot,Add,Subtract,GaussianDropout
from keras.utils import Sequence,plot_model
from keras.models import Model,load_model
from keras import backend as K


# ### Denoising Autoencoder with weak supervision

# In[94]:


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


# In[95]:


x=Input(shape=(5,))
y=Lambda(noisefunction)(x)
model=Model(x,y)
x=np.arange(2*5).reshape((2,5))
model.predict(x)


# In[197]:


K.clear_session()
np.random.seed(100)
###############
# instantiate layer
# noise=Lambda(noisefunction,name="noisefunction")
# trilosslayer=Lambda(Triplet,name="trilosslayer")
# dense=Dense(units=12000,activation="sigmoid",name="Dense1")
###############
#parameter setting
BOW_dim=20000
DR_dim=100
loss_weights=[1,1,1,1]

##Encoder
x=Input((BOW_dim,),name="encoder_input")
# y=Lambda(noisefunction,name="noisefunction")
y=GaussianDropout(rate=0.2,name="noise")(x)
# y=Dense(30,use_bias=False)(y)
y=Dense(units=DR_dim,activation="sigmoid",name="Dense",use_bias=False)(y)
encoder=Model(x,y,name="encoder")

##Decoder
x=Input((DR_dim,),name="input") 
y=Dense(units=20000,activation="sigmoid",name="Dense")(x)
decoder=Model(x,y,name="decoder")

##Tripletloss
x0=Input((DR_dim,),name="anchor_input")
x1=Input((DR_dim,),name="positive_input")
x2=Input((DR_dim,),name="negative_input")
r01=Dot(axes=1,name="SimiPositive")([x0,x1])
r02=Dot(axes=1,name="SimiNegative")([x0,x2])
out=Lambda(tripletlossfunction,name="tripletlossfunction")([r01,r02])
tripletloss=Model(inputs=[x0,x1,x2],outputs=out,name="triplet")

#Build Tri-Autoencoder model
x0=Input((BOW_dim,),name="anchor_input")
x1=Input((BOW_dim,),name="positive_input")
x2=Input((BOW_dim,),name="negative_input")

h0=encoder(x0)
h1=encoder(x1)
h2=encoder(x2)

Lt=tripletloss([h0,h1,h2])

y0=decoder(h0)
y1=decoder(h1)
y2=decoder(h2)

y0=Lambda(lambda x :x ,name="anchor")(y0)
y1=Lambda(lambda x :x ,name="positive")(y1)
y2=Lambda(lambda x :x ,name="negative")(y2)

# triplet_loss

Tri_AutoEncoder=Model(inputs=[x0,x1,x2],outputs=[y0,y1,y2,Lt])
Tri_AutoEncoder.compile(optimizer="adam",
                        loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy',losspassfunction],
                        loss_weights=loss_weights
                       )
#save initial model
Tri_AutoEncoder.save("Tri_AutoEncoder.initial.h5")
encoder.save("encoder.initial.h5")
decoder.save("decoder.initial.h5")
#plot
plot_model(Tri_AutoEncoder,to_file="Tri_DenoiseAutoEncoder.png")
plot_model(encoder,to_file="encoder.png")
plot_model(decoder,to_file="decoder.png")
# Open the file to record
with open('Tri_AutoEencoderncoder.summary.txt','w') as f:
    # Pass the file handle in as a lambda function to make it callable
    Tri_AutoEncoder.summary(print_fn=lambda x: f.write("    "+x + '\n'))
with open('encoder.summary.txt','w') as f:
    # Pass the file handle in as a lambda function to make it callable
    encoder.summary(print_fn=lambda x: f.write("    "+x + '\n'))
with open('decoder.summary.txt','w') as f:
    # Pass the file handle in as a lambda function to make it callable
    decoder.summary(print_fn=lambda x: f.write("    "+x + '\n'))

Tri_AutoEncoder.summary()


# ## Load Data

# In[6]:


# Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[7]:


# with open("D:3.AutoencoderForArticle/train_dict_collect_small_industry","rb") as f:
#     train_dict_collect_small_industry=pickle.load(f)
# with open("D:3.AutoencoderForArticle/train_dict_small_triplet_v02","rb") as f:
#     train_dict_small_triplet_v02=pickle.load(f)
# with open("D:3.AutoencoderForArticle/test_dict_collect_small_industry","rb") as f:
#     test_dict_collect_small_industry=pickle.load(f)
# with open("D:3.AutoencoderForArticle/test_dict_small_triplet_v02","rb") as f:
#     test_dict_small_triplet_v02=pickle.load(f)


# In[8]:


#train treiplet
for i,k in enumerate(train_dict_small_triplet_v02):
    if i==0:train_tripletindex=train_dict_small_triplet_v02[k]
    else:train_tripletindex=np.concatenate((train_tripletindex,train_dict_small_triplet_v02[k]),axis=0)


# In[9]:


len(train_tripletindex)


# In[10]:


total=0
for k in train_dict_small_triplet_v02:
    total+=len(train_dict_small_triplet_v02[k])
total


# In[11]:


#test treiplet
for i,k in enumerate(test_dict_small_triplet_v02):
    if i==0:test_tripletindex=test_dict_small_triplet_v02[k]
    else:test_tripletindex=np.concatenate((test_tripletindex,test_dict_small_triplet_v02[k]),axis=0)


# In[12]:


len(test_tripletindex)


# In[13]:


total=0
for k in test_dict_small_triplet_v02:
    total+=len(test_dict_small_triplet_v02[k])
total


# In[ ]:


# %%time
# np.random.permutation(train_tripletindex)


# In[ ]:


# %%time
# np.random.shuffle(train_tripletindex)


# In[ ]:


# %%time
# pick=np.random.permutation(181054122)
# train_tripletindex[pick]


# In[20]:


get_ipython().run_cell_magic('time', '', 'b=32\nbatch_index = train_tripletindex[0 * b:(0 + 1) * b]\npprint(batch_index)')


# ## Data generator

# In[139]:


class DataGenerator(Sequence):

    def __init__(self, tripletindex,Data, batch_size=32):
        self.tripletindex= np.asarray(tripletindex)
        pick=np.random.permutation(len(self.tripletindex))
        self.tripletindex= self.tripletindex[pick]
        self.batch_size = batch_size

    def __len__(self):
#         return int(np.ceil(self.tripletindex.shape[0] / float(self.batch_size)))
        return 100
    def __getitem__(self, idx):
        batch_index = self.tripletindex[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_anchor = Data[batch_index[:,0]]
        batch_positive = Data[batch_index[:,1]]
        batch_negative = Data[batch_index[:,2]]
        tripletloss=np.zeros((batch_negative.shape[0],1))
        return [batch_anchor,batch_positive,batch_negative],[batch_anchor,batch_positive,batch_negative,tripletloss]
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        
#         pick=np.random.permutation(len(self.tripletindex))
#         self.tripletindex= self.tripletindex[pick]
        


# In[140]:


get_ipython().run_cell_magic('time', '', '# train0=np.random.randint(0,2,(4,12000))\n# train1=np.random.randint(0,2,(4,12000))\n# train2=np.random.randint(0,2,(4,12000))\n# labels=np.ones((4,1))\n\n\n#setup\n\nbatch_size=32\n#instantiate generator\ntraingenerator=DataGenerator(tripletindex=train_tripletindex,\n                             Data=Data,\n                             batch_size=batch_size)\ntestgenerator=DataGenerator(tripletindex=test_tripletindex,\n                            Data=Data,\n                            batch_size=batch_size)')


# In[173]:


get_ipython().run_cell_magic('time', '', '#setup\nepochs=200\n# steps_per_epoch=5\n#train\nHistory=Tri_AutoEncoder.fit_generator(generator=traingenerator,\n                                      shuffle=True,\n                                      epochs=epochs,\n#                                       steps_per_epoch=steps_per_epoch,\n                                      validation_data=testgenerator,\n                                      verbose=2,\n                                      workers=3,use_multiprocessing=False,\n                                    \n                                     )')


# In[186]:


Tri_AutoEncoder.save("Tri_AutoEncoder_trained.h5")
encoder.save("encoder_trained.h5")
decoder.save("decoder_trained.h5")


# In[175]:


df=pd.DataFrame(History.history)


# In[188]:


df.to_hdf("history.h5",key="data")


# In[189]:


# df=pd.read_hdf("history.h5")


# In[190]:


# df[["loss","val_loss"]].plot(subplots=False,layout=(1,3),figsize=(18,6))


# In[191]:


df[["loss","val_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[192]:


df[["triplet_loss","val_triplet_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[193]:


df[["anchor_loss","positive_loss","negative_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[194]:


df[["val_anchor_loss","val_positive_loss","val_negative_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[195]:


df.to_hdf("history.h5",key="data")


# In[196]:


pd.read_hdf("history.h5")


# In[199]:


import keras
print(keras.__version__)

