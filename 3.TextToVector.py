
# coding: utf-8

# In[119]:


from ArticlesRep import MeanSimilarity

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
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'
list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]


# ## load Dataset

# In[4]:


with open("D:3.AutoencoderForArticle/DataSet_vip_single_industry","rb") as f:
    DataSet=pickle.load(f)


# In[5]:


len(DataSet)


# In[5]:


DataSet.info()


# In[6]:


_=DataSet["title_token_plus_body_token"]
_=_.value_counts()
_.reset_index().rename(columns={"index":"title_token_plus_body_token","title_token_plus_body_token":"counts"})


# In[310]:


# with open("D:3.AutoencoderForArticle/testset_vip.v01.p","wb") as f:
#     pickle.dump(file=f,obj=trainset_vip)


# In[311]:


# with open("D:3.AutoencoderForArticle/testset_vip.v01.p","rb") as f:
#     x=pickle.load(f)


# ### Vectorization

# In[21]:


from keras.preprocessing.text import Tokenizer


# In[62]:


texts = DataSet["title_token_plus_body_token"]
tokenizer = Tokenizer(num_words=20000,)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# word_index


# In[29]:


_=pd.Series(tokenizer.word_counts)
_.sort_values(ascending=False,inplace=True)


# In[61]:


_[:20000][[0,1,2,3,4,5,-6,-5,-4,-3,-2,-1]]


# In[68]:


_=pd.Series(tokenizer.word_docs)
_.sort_values(ascending=False,inplace=True)
_[:20000][[0,1,2,3,4,5,-6,-5,-4,-3,-2,-1]]


# In[70]:


one_hot_results = tokenizer.texts_to_matrix(texts, mode='binary')
one_hot_results


# In[88]:


one_hot_results=one_hot_results.astype(int)


# In[90]:


one_hot_results


# ### Save one_hot_results 

# In[91]:


# np.save("D:3.AutoencoderForArticle/BOW_binary_v01.npy",one_hot_results)


# #### Load BOW_binary_v01

# In[6]:


BOW_binary=np.load("D:3.AutoencoderForArticle/BOW_binary_v01.npy")


# In[7]:


#sparcity
sparcity=np.mean(BOW_binary)
print("{}%".format(np.round(sparcity,5)*100))


# In[8]:


np.sum(BOW_binary[69520])


# ## Build Model

# In[138]:


from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras.losses import mean_absolute_error
input_dim=20000
hidden_dim=500
input_text = Input(shape=(input_dim,),name="Input")
# "encoded" is the encoded representation of the input
encoded = Dense(hidden_dim, activation='relu',name="Encoder")(input_text)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid',name="Decoder")(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs=input_text, outputs=decoded)
#compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=[mean_absolute_error])
# autoencoder.compile(optimizer='adadelta', loss='mse')
#summary
autoencoder.summary()
#save initial weights
autoencoder.save('autoencoder.initial.h5')


# In[19]:


#plot model
from keras import utils
utils.plot_model(autoencoder, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')


# In[130]:


# this model maps an input to its encoded representation
encoder = Model(inputs=input_text, outputs=encoded)
encoder.summary()


# In[314]:


# create a placeholder for an encoded input
encoded_input = Input(shape=(hidden_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
decoder.summary()


# ### TrainSet and TestSet prepare

# In[20]:


# random pick 10000 pieces
seed=5
train_num=10000
vali_num=2000
total=DataSet.shape[0]
np.random.seed(seed)
picksample=np.random.permutation(total)[:train_num+vali_num]
print(picksample)
__ = DataSet.iloc[picksample]["indusrty_tags"]


# In[139]:


#Save picksample
# picksample=np.save("D:3.AutoencoderForArticle/picksample",picksample)


# In[21]:


dict_={k:0 for k in list_industry}
for t in tqdm_notebook(__):
#     print(t)
    for i in dict_:
        if i in t:dict_[i]+=1
ser=pd.Series(dict_)
ser.sort_values(ascending=False)


# In[22]:


x_train=BOW_binary[picksample[:train_num]]


# In[23]:


x_test=BOW_binary[picksample[train_num:train_num+vali_num]]


# # Metric before Train

# In[24]:


metrics_train=np.mean(np.absolute(x_train[0:1]-autoencoder.predict(x_train[0:1])))
metrics_test=np.mean(np.absolute(x_test-autoencoder.predict(x_test)))
metrics_train_on_zeros=np.mean(np.absolute(x_train-np.zeros_like(x_train)))
metrics_test_on_zeros=np.mean(np.absolute(x_test-np.zeros_like(x_test)))
print(metrics_train)
print(metrics_test)
print(metrics_train_on_zeros)
print(metrics_test_on_zeros)


# ## Train  

# In[337]:


from keras_tqdm import TQDMNotebookCallback
#callback
checkpointer = ModelCheckpoint(filepath="best_model_autoencoder.hdf5",save_weights_only=False,
                               monitor='val_loss',verbose=1, save_best_only=True,period=5)
#fit
epochs=1
batch_size=526
history=autoencoder.fit(x_train, x_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=0,
                        callbacks=[TQDMNotebookCallback(),
#                                    checkpointer,
                                  ],
#                         validation_data=(x_test, x_test),
                        
                       )


# In[104]:


autoencoder.predict(x_train_small[0:1])


# In[99]:


df=pd.DataFrame(autoencoder.history.history)
df["mean_absolute_error"][:].plot()


# In[44]:


df[10:].plot()


# In[47]:


df[:20].plot()


# # Metrics Mean Square Error

# In[133]:


from keras.models import load_model
autoencoder=load_model("best_weights.hdf5")


# In[134]:


metrics_train=np.mean(np.absolute(x_train-autoencoder.predict(x_train)))
metrics_test=np.mean(np.absolute(x_test-autoencoder.predict(x_test)))
metrics_train_on_zeros=np.mean(np.absolute(x_train-np.zeros_like(x_train)))
metrics_test_on_zeros=np.mean(np.absolute(x_test-np.zeros_like(x_test)))
print(metrics_train)
print(metrics_test)
print(metrics_train_on_zeros)
print(metrics_test_on_zeros)


# # Similary Encoded_vector

# In[28]:


picksample


# ### build encoder

# In[135]:


from keras.layers import Input
from keras.models import Model
encoder_layer=autoencoder.layers[1]
# Data flow
input_x=Input((20000,))
out=encoder_layer(input_x)
encoder = Model(inputs=input_x, outputs=out)


# In[30]:


x_encoded=encoder.predict(x_train)


# In[31]:


simi=cosine_similarity(x_encoded)


# In[50]:


simi_argsort=np.argsort(simi*(-1))
simi_argsort


# In[35]:


DataSet.iloc[picksample[simi_argsort][0]]


# In[37]:


DataSet.iloc[picksample[simi_argsort][np.where(picksample==14047)[0][0]]]


# In[39]:


DataSet.iloc[picksample[simi_argsort][np.where(picksample==61956)[0][0]]]


# In[40]:


DataSet.iloc[picksample[simi_argsort][np.where(picksample==32783)[0][0]]]


# In[45]:


DataSet.iloc[picksample[simi_argsort][np.where(picksample==93656)[0][0]]]


# ## Similarity in same industry

# In[66]:


picksample


# In[67]:


# picksample[picksample==dict_collect_industry["能源"][0]]
df=DataSet.iloc[picksample]


# In[68]:


df


# # Similary original_vector

# In[51]:


simi_2=cosine_similarity(x_train)


# In[52]:


simi_argsort_2=np.argsort(simi_2*(-1))
simi_argsort_2


# In[54]:


DataSet.iloc[picksample[simi_argsort_2][np.where(picksample==112672)[0][0]]]


# In[ ]:


dict_collect_industry[]


# In[58]:


DataSet.iloc[picksample[simi_argsort_2][np.where(picksample==32783)[0][0]]]


# In[59]:


DataSet.iloc[picksample[simi_argsort_2][np.where(picksample==61956)[0][0]]]


# In[60]:


DataSet.iloc[picksample[simi_argsort_2][np.where(picksample==14047)[0][0]]]


# ## Mean Similarity in same industry

# In[72]:


get_ipython().run_cell_magic('time', '', 'dict_collect_industry={k:[] for k in list_industry}\nfor i,r in df[:].iterrows():\n#     print(i)\n    dict_collect_industry[list(r["indusrty_tags"])[0]].append(i)')


# In[144]:


# with open("D:3.AutoencoderForArticle/dict_collect_industry.p","wb") as f:
#     pickle.dump(file=f,obj=dict_collect_industry,protocol=True)


# In[73]:


len(dict_collect_industry["傳播出版"])


# In[76]:


dict_collect_industry["傳播出版"]


# In[123]:


len(BOW_binary[dict_collect_industry["傳播出版"]])


# In[124]:


arr=BOW_binary[dict_collect_industry["傳播出版"]]


# In[125]:


MeanSimilarity(arr)


# In[136]:


arr_encoded=encoder.predict(arr)
arr_encoded.shape


# In[137]:


MeanSimilarity(arr_encoded)


# In[141]:


pd.DataFrame(dict_collect_industry)

