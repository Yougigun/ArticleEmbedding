
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
import os
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


# In[2]:


from keras.layers import Dense,Lambda,Input,Dot,Add,Subtract,GaussianDropout
from keras.utils import Sequence,plot_model
from keras.models import Model,load_model
from keras import backend as K


# ### Denoising Autoencoder with weak supervision

# In[3]:


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


# In[4]:


x=Input(shape=(5,))
y=Lambda(noisefunction)(x)
model=Model(x,y)
x=np.arange(2*5).reshape((2,5))
model.predict(x)


# In[5]:


K.clear_session()
np.random.seed(100)
###############
# instantiate layer
# noise=Lambda(noisefunction,name="noisefunction")
# trilosslayer=Lambda(Triplet,name="trilosslayer")
# dense=Dense(units=12000,activation="sigmoid",name="Dense1")
###############
#model path
path="Models/Model3_on_all_industry"
if not os.path.isdir(path):
    os.mkdir(path)

#parameter setting    
BOW_dim=19404
DR_dim=100
loss_weights=[1,1,1,2]

##Encoder
x=Input((BOW_dim,),name="encoder_input")
y=GaussianDropout(rate=0.2,name="noise")(x)
# y=Dense(units=2000,activation="sigmoid",name="Dense_1")(y)
y=Dense(units=DR_dim,activation="sigmoid",name="Dense_1",use_bias=False)(y)
encoder=Model(x,y,name="encoder")

##Decoder
x=Input((DR_dim,),name="input") 
y=x
y=Dense(units=BOW_dim,activation="sigmoid",use_bias=True,name="Dense_1")(y)
# y=Dense(units=BOW_dim,activation="sigmoid",name="Dense_2")(y)
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
Tri_AutoEncoder.save("{}/Tri_AutoEncoder.initial.h5".format(path))
# encoder.save("encoder.initial.h5")
# decoder.save("decoder.initial.h5")
#plot
plot_model(Tri_AutoEncoder,to_file="{}/Tri_DenoiseAutoEncoder.png".format(path))
plot_model(encoder,to_file="{}/encoder.png".format(path))
plot_model(decoder,to_file="{}/decoder.png".format(path))
# Open the file to record
with open('{}/Tri_AutoEencoderncoder.summary.txt'.format(path),'w') as f:
    # Pass the file handle in as a lambda function to make it callable
    Tri_AutoEncoder.summary(print_fn=lambda x: f.write("    "+x + '\n'))
with open('{}/encoder.summary.txt'.format(path),'w') as f:
    # Pass the file handle in as a lambda function to make it callable
    encoder.summary(print_fn=lambda x: f.write("    "+x + '\n'))
with open('{}/decoder.summary.txt'.format(path),'w') as f:
    # Pass the file handle in as a lambda function to make it callable
    decoder.summary(print_fn=lambda x: f.write("    "+x + '\n'))

Tri_AutoEncoder.summary()


# ## Load Data

# In[6]:


Data=np.load("D:3.AutoencoderForArticle/BOW_binary_v02.npy")


# In[7]:


with open("D:3.AutoencoderForArticle/train_dict_collect_industry_50000.p","rb") as f:
    train_dict_collect_industry_50000=pickle.load(f)


# In[8]:


train_dict_collect_industry_50000.pop("控股","沒有")


# In[9]:


with open("D:3.AutoencoderForArticle/test_dict_collect_industry_50000.p","rb") as f:
    test_dict_collect_industry_50000=pickle.load(f)


# In[10]:


test_dict_collect_industry_50000.pop("控股","沒有")


# ## Data generator

# In[11]:


class DataGenerator(Sequence):

    def __init__(self, tripletindex,Data, batch_size=32):
        self.tripletindex= np.asarray(tripletindex)
        pick=np.random.permutation(len(self.tripletindex))
        self.tripletindex= self.tripletindex[pick]
        self.batch_size = batch_size
        self.Data=Data

    def __len__(self):
         return int(np.ceil(self.tripletindex.shape[0] / float(self.batch_size)))
#         return 1
    def __getitem__(self, idx):
        batch_index = self.tripletindex[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_anchor = self.Data[batch_index[:,0]]
        batch_positive = self.Data[batch_index[:,1]]
        batch_negative = self.Data[batch_index[:,2]]
        tripletloss=np.zeros((batch_negative.shape[0],1))
        return [batch_anchor,batch_positive,batch_negative],[batch_anchor,batch_positive,batch_negative,tripletloss]
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pick=np.random.permutation(len(self.tripletindex))
        self.tripletindex= self.tripletindex[pick]
        


# In[12]:


class HardTriGenerator(Sequence):

    def __init__(self, dict_id_news,Data, P=3,K=3):
        self.dict_id_news=dict_id_news
        self.industry=np.asarray(list(dict_id_news.keys()))
#         pick=np.random.permutation(len(self.tripletindex))
#         self.tripletindex= self.tripletindex[pick]
        self.P=P
        self.K=K
        self.Data=Data

    def __len__(self):
        return int(np.floor(len(self.industry) / self.P))
#         return 1
    def __getitem__(self, idx):
        indusrty= self.industry[idx*self.P:(idx+1)*self.P]
        small_dict_id_news={i:np.random.choice(self.dict_id_news[i],size=self.K,replace=False) for i in indusrty}
        dict_small_triplet=dict()
        for k in small_dict_id_news:
            poslist=small_dict_id_news[k]
            poslen=len(poslist)
            neglist=[]
            for j in small_dict_id_news:
                if k!=j:neglist+=list(small_dict_id_news[j])
                neglen=len(neglist)
            indarray=np.zeros((int(poslen*(poslen-1)*neglen),3),dtype=int)
            i=0
            for ip1_,a in enumerate(poslist):
                for ip2_,p in enumerate(poslist):
                    if ip1_!=ip2_:
                        for in_,n in enumerate(neglist):
        #                   print(p1,p2,n)
                            indarray[i,0]=a
                            indarray[i,1]=p
                            indarray[i,2]=n
                            i+=1
        #                 break
        #             break
        #         break
            dict_small_triplet[k]=indarray            
        for i,k in enumerate(dict_small_triplet):
            if i==0:tripletindex=dict_small_triplet[k]
            else:tripletindex=np.concatenate((tripletindex,dict_small_triplet[k]),axis=0)   
        batch_anchor=self.Data[tripletindex[:,0]]
        batch_positive=self.Data[tripletindex[:,1]]
        batch_negative=self.Data[tripletindex[:,2]]
        tripletloss=np.zeros((batch_negative.shape[0],1))    
        
        return [batch_anchor,batch_positive,batch_negative],[batch_anchor,batch_positive,batch_negative,tripletloss]
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.industry=np.random.permutation(self.industry)       


# In[17]:


#setup

#instantiate generator
traingenerator=HardTriGenerator(dict_id_news=train_dict_collect_industry_50000,
                                Data=Data,
                                P=3, # remainder>=2;
                                K=5,
                               )
testgenerator=HardTriGenerator(dict_id_news=test_dict_collect_industry_50000,
                               Data=Data,P=3,K=5,
                              )


# In[18]:


for _,i in enumerate(traingenerator):
    print(i[0][2].shape)
    break


# ## Callback function

# In[19]:


from keras.callbacks import TerminateOnNaN,ModelCheckpoint,TensorBoard
#creat regular folder

regular_path =path+"/regular"
logname="model3allindustry50000"
if not os.path.isdir(regular_path):
    os.mkdir(regular_path)
# Instantiation claaback function
checkpointer = ModelCheckpoint(filepath='{}/bestmodel.hdf5'.format(path), verbose=0, save_best_only=True,period=10)
regularsave = ModelCheckpoint(filepath="{}".format(path)+'/regular/weights.{epoch:02d}.hdf5',
                              save_weights_only=True, 
                              verbose=0,
                              save_best_only=False,period=50)

tensorboard=TensorBoard(log_dir="./logs/{}".format(logname))


# ## Train

# In[21]:


#initial
Tri_AutoEncoder=load_model("{}/Tri_AutoEncoder.initial.h5".format(path),custom_objects={"losspassfunction":losspassfunction})
#setup
epochs=4560
#train
History=Tri_AutoEncoder.fit_generator(callbacks=[checkpointer,tensorboard,regularsave],
                                      generator=traingenerator,
#                                       shuffle=True,
                                      epochs=epochs,
#                                       steps_per_epoch=steps_per_epoch,
                                      validation_data=testgenerator,
                                      verbose=2,
                                      workers=1,use_multiprocessing=False,
                                      
                                    
                                     )
#save model
Tri_AutoEncoder.save("{}/Tri_AutoEncoder_trained.h5".format(path))
Tri_AutoEncoder.save_weights("{}/weights.h5".format(path))


# In[19]:


# #partial fit
# Tri_AutoEncoder=load_model("Tri_AutoEncoder.initial.h5",custom_objects={"losspassfunction":losspassfunction})
# epoch=1
# for i in range(epoch):
#     for i in traingenerator:
#         Tri_AutoEncoder.train_on_batch(x=i[0],y=i[1])


# In[22]:


df=pd.DataFrame(History.history)
df.to_hdf("{}/history.h5".format(path),key="data")


# In[23]:


df[["loss","val_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[24]:


df[["triplet_loss","val_triplet_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[25]:


df[["anchor_loss","positive_loss","negative_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))


# In[26]:


df[["val_anchor_loss","val_positive_loss","val_negative_loss"]].plot(subplots=True,layout=(1,3),figsize=(18,6))

