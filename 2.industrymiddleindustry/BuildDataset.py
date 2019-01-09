
# coding: utf-8

# In[43]:


import pickle
import numpy as np
import pandas as pd
import uuid
industry=['金融業', 'IC製造', 'IC設計', '電子零件元件', '電腦系統業', '石油及天然氣', '手機', '面板業', '太陽能',
       '板鋼', '被動元件', '車輛整車', '非鐵金屬', '流通業', '地產', '通訊設備', '遊戲產業', '印刷電路板相關',
       '貴金屬', '機械', '運輸事業', '電子通路', 'IC封裝測試', 'LED', '通訊服務', '消費性電子產品', '化學工業',
       '醫藥產業', '軟體業', '週邊產品', '石化業', '旅館、餐飲', '汽機車零組件', '設備儀器廠商', '條鋼', '造紙業',
       '水泥', '橡膠工業', '傳產其他', '數位相機', '顯示器', '不鏽鋼', '農林漁牧', '服務業', 'Internet相關',
       '營造工程', '休閒娛樂', '紡織中游', '生物科技', '家電', '光碟片', '建材', '成衣', '電力', '運動產業',
       '線材、盤元', '化纖原料', '其他公用事業', '大宗物資', '家居用品', '手機零組件', '食品加工', '電力設備',
       '航天軍工', '分離式元件', '電子其他', '面板零組件', '飲料相關', '電線電纜', '封測服務與材料', '礦石開採',
       '光通訊', '傳播事業', '輔助與彌補用醫材', '時尚產業', '基礎建設營運', '電聲產品', '水資源', '醫療器材通路',
       '合金鋼', '文化創意產業', '車用金屬成型', '穿戴式裝置', '其他醫療器材', '汽車內裝', '車用電子',
       '診斷與監測用醫材', '傳輸介面', '電子化工材料', '電池材料相關', '資產股', '醫療管理服務', '體外診斷用醫材', '煤',
       '射頻前端晶片', '控股公司', '無店舖販售', '手術與治療用醫材', '金屬礦採選', '生物辨識相關']
from torch.utils.data import Dataset


# In[267]:


class tripletdataset(Dataset):
    def __init__(self,industry,dict_industry_guid,dict_guid_bow,P=3,K=8):
        self.industry=industry
        self.dict_industry_guid=dict_industry_guid
        self.dict_guid_bow=dict_guid_bow
        self.P=P
        self.K=K
    def __len__(self):
        return int(np.floor(len(self.industry) / self.P))
    
    def __getitem__(self,idx):
        industrys=self.industry[idx*self.P:(idx+1)*self.P]
        
        small_dict_id_news={i:np.random.choice(self.dict_industry_guid[i],size=self.K,replace=False) for i in industrys}
        
        dict_small_triplet=dict()
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
        return (batch_anchor,batch_positive,batch_negative)
            
    def shuffle(self):
        self.industry=np.random.permutation(np.asarray(self.industry))
            
        


# In[287]:


if __name__=="__main__":
    with open("D:4.AutoencoderForArticle.Mid-Industry/dict_singleindustry_guid.p","rb" ) as f:
        dict_singleindustry_guid=pickle.load(f)
    with open("D:4.AutoencoderForArticle.Mid-Industry/dict_guid_sparsebow_single.p","rb" ) as f:
        dict_guid_sparsebow_single=pickle.load(f)
        
    dataset=tripletdataset(industry[:-20],dict_singleindustry_guid,dict_guid_sparsebow_single,P=3,K=5)

    dataset.shuffle()

    dataset[0]

