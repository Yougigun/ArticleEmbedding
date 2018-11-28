
# coding: utf-8

# In[1]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[2]:


def MeanSimilarityoneindustry(array):
    array=np.asarray(array)
    simi_=cosine_similarity(array)
    simi_tri=np.triu(simi_,1)
    sum_=np.sum(simi_tri)
    n=((1+array.shape[0]-1)*(array.shape[0]-1))/2
    mean=sum_/n
    return mean


# In[8]:


def MeanSimilaritytwoindustry(array1,array2):
    arra1=np.asarray(array1)
    arra2=np.asarray(array2)
    simi_=cosine_similarity(arra1,arra2)
    return np.mean(simi_)

