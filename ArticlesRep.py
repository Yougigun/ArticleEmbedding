
# coding: utf-8

# In[1]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[2]:


def MeanSimilarityoneindustry(array,metric="cosine"):
    array=np.asarray(array)
    if metric=="cosine":
        simi=cosine_similarity(array)
    else:
        simi=np.zeros((array.shape[0],array.shape[0]))
        for i,u in enumerate(array):
            for j,v in enumerate(array):
                simi[i,j]=metric(u,v)
    simi_tri=np.triu(simi_,1)
    sum_=np.sum(simi_tri)
    n=((1+array.shape[0]-1)*(array.shape[0]-1))/2
    mean=sum_/n
    return mean


# In[8]:


def MeanSdSimilaritytwoindustry(array1,array2,metric="cosine"):
    arra1=np.asarray(array1)
    arra2=np.asarray(array2)
    if metric=="cosine":
        simi_=cosine_similarity(arra1,arra2)
    else:
        simi=np.zeros((arra1.shape[0],arra2.shape[0]))
        for i,u in enumerate(array):
            for j,v in enumerate(array):
                simi[i,j]=metric(u,v)
    return np.mean(simi_)


# In[ ]:


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

