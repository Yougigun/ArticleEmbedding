
# coding: utf-8

# In[43]:


import numpy as np
import uuid
import torch
from torch.utils.data import Dataset,DataLoader


# In[300]:


class Tripletdataset(Dataset):
    def __init__(self,industry,dict_industry_guid,dict_guid_bow,P=3,K=8):
        self.industry=industry
        self.dict_industry_guid=dict_industry_guid
        self.dict_guid_bow=dict_guid_bow
        self.P=P
        self.K=K
    def __len__(self):
        return int(np.floor(len(self.industry) / self.P))
    
    def __getitem__(self,idx):
        if len(self)==idx: raise StopIteration
        industrys=self.industry[idx*self.P:(idx+1)*self.P]
        small_dict_id_news={i:np.random.choice(self.dict_industry_guid[i],size=self.K,replace=False) for i in industrys}
        
        dict_small_triplet=dict()
        #### PK x (K-1) x (PK-K)
        for k in small_dict_id_news:
            poslist=small_dict_id_news[k]
            poslen=len(poslist)
            neglist=[]
            for j in small_dict_id_news:
                if k!=j:neglist+=list(small_dict_id_news[j])
            neglen=len(neglist)
            indarray=np.empty((int(poslen*(poslen-1)*neglen),3),dtype='U36')
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
            dict_small_triplet[k]=indarray  
        for i,k in enumerate(dict_small_triplet):
            if i==0:tripletindex=dict_small_triplet[k]
            else:tripletindex=np.concatenate((tripletindex,dict_small_triplet[k]),axis=0)             
        
        batch_anchor=np.asarray([self.dict_guid_bow[uuid.UUID(v)].toarray() for v in tripletindex[:,0]]).squeeze()
        batch_positive=np.asarray([self.dict_guid_bow[uuid.UUID(v)].toarray() for v in tripletindex[:,1]]).squeeze()
        batch_negative=np.asarray([self.dict_guid_bow[uuid.UUID(v)].toarray() for v in tripletindex[:,2]]).squeeze()
        return {"batch_anchor":batch_anchor,"batch_positive":batch_positive,"batch_negative":batch_negative}
            
    def shuffle(self):
        self.industry=np.random.permutation(np.asarray(self.industry))
            
        


# In[287]:


if __name__=="__main__":
    import pickle
    from scipy import sparse
    with open("D:4.AutoencoderForArticle.Mid-Industry/dict_singleindustry_guid.p","rb" ) as f:
        dict_singleindustry_guid=pickle.load(f)
    with open("D:4.AutoencoderForArticle.Mid-Industry/dict_guid_sparsebow_single.p","rb" ) as f:
        dict_guid_sparsebow_single=pickle.load(f)
        
    dataset=tripletdataset(industry[:-20],dict_singleindustry_guid,dict_guid_sparsebow_single,P=3,K=5)

    dataset.shuffle()

    dataset[0]

