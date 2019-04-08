import torch
from torch.utils.data import Dataset , DataLoader
from torch.nn.utils.rnn import pack_sequence
import numpy as np
import pandas as pd
class UsersReprzRNN(torch.nn.Module):
    def __init__(self,articleEmbeddingSize=None,userEmbeddingSize=None,
                 batch_first=True):
        super(UsersReprzRNN,self).__init__()
        self.articleEmbeddingSize=articleEmbeddingSize
        self.userEmbeddingSize=userEmbeddingSize
        self.batch_first=batch_first
        self.gru =  torch.nn.GRU(self.articleEmbeddingSize,
                                 self.userEmbeddingSize,
                                 batch_first=batch_first)
        self.linear=torch.nn.Linear(in_features=self.userEmbeddingSize,
                                    out_features=self.userEmbeddingSize)
        self.tanh = torch.nn.Tanh()
    def forward(self,browsing_history= None):
        x=browsing_history
        _,x=self.gru(x)
        x=self.linear(x)
        y=self.tanh(x)
        return y


class UserDataset(Dataset):
    def __init__(self, x,y) :
        assert all(x.size(0) == y.size(0))
        self.x=x
        self.y=y
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self,index):
        return tuple(self.x[index],self.y[index])


class UserRepzTrainer():
    def __init__(self,trainset=None,target=None, vis=None ,):
        # self.epoch_start=epoch_start
        # self.epoch = epoch
        self.trainset=trainset
#        self.optimzer=optimizer
        self.vis=vis

    def metrics(self,):
        pass

class Dataiterator():
    def __init__(self,dataset,batch_size=32):
        self.batch_size=batch_size
        self.dataset=np.random.permutation(dataset)
        self.last=len(self)
        self.n=0
    def shuffle(self):
        self.dataset=np.random.permutation(self.dataset)
    def __iter__(self):
        return self
    def __len__(self):
        return int(np.floor(len(self.dataset)/self.batch_size))
    def __next__(self):
        if self.last==self.n :
            self.shuffle()
            self.n=0
            raise StopIteration
        self.batch=self.dataset[self.n*self.batch_size:(self.n+1)*self.batch_size]
        self.n+=1
        return self.batch




class DataPreprocess():
    @staticmethod
    def splitbyuser(dataset):
        DataSetSplitbyUser=dict()
        for sess in dataset:
            if sess["User"] not in DataSetSplitbyUser :
                DataSetSplitbyUser[sess["User"]]=[sess]
            else:
                DataSetSplitbyUser[sess["User"]].append(sess)
        return DataSetSplitbyUser


    @staticmethod
    def splitbytime(ModelSet,timepoint):
        assert isinstance(timepoint,str) ,"timepoint must be string"
        timepoint=pd.Timestamp(timepoint)
        ModelSetSplitbytime={"Before":[],"After":[]}
        for sess in ModelSet:
            if sess["Date"]>=timepoint : ModelSetSplitbytime["After"].append(sess)
            else :  ModelSetSplitbytime["Before"].append(sess)
        return ModelSetSplitbytime
    @staticmethod
    def modelsetsplit(ModelSet,split=6/7):
        ModelSet=np.random.permutation(ModelSet)
        TrainSet=ModelSet[:int(np.floor(len(ModelSet)*split))]
        ValiSet=ModelSet[int(np.floor(len(ModelSet)*split)):]
        return (TrainSet,ValiSet)
    @staticmethod
    def extractlog(sess):
        return sess["Data"]["LogReprz"]
    @staticmethod
    def batchLogProcess(batch,device="cpu"):
        logLength=[len(sess["Data"]["Log"]) for sess in batch]
        logLengthArgSort=np.argsort(logLength)[::-1]
        return pack_sequence([torch.from_numpy(batch[ind]["Data"]["LogReprz"])
                              for ind in logLengthArgSort]).to(device=device)
    @staticmethod
    def batchSessProcess(batch,device="cpu"):
        logLength=[len(sess["Data"]["Log"]) for sess in batch]
        logLengthArgSort=np.argsort(logLength)[::-1]
        return [ {"SessReprz":torch.from_numpy(
                                            batch[ind]["Data"]["SessReprz"]
                                            ).to(device=device),
                "Click":batch[ind]["Data"]["Session"]["Click"],
                "Unclick":batch[ind]["Data"]["Session"]["Unclick"],
                 "Date":batch[ind]["Date"],
                  "User":batch[ind]["User"],
                  "PosInList":batch[ind]["PosInList"],
                  "Log":batch[ind]["Data"]["Log"],
                  "List":batch[ind]["Data"]["Session"]["List"],
                  
                 }
                for ind in logLengthArgSort  ]
    @classmethod
    def batchProcess(cls,batch,device="cpu"):
        return cls.batchLogProcess(batch,device),cls.batchSessProcess(batch,
                                                                      device)

class TriLossOri():
    def __init__(self,reduction="mean",size=None):
        self.reduction=reduction.lower()
        self.size=size
    def forward(self,h0,h1,h2):
        h01=(h0*h1).sum(dim=1)
        h02=(h0*h2).sum(dim=1)
        Lt=torch.log(1+torch.exp(h02-h01))
        if self.reduction=="mean":
            assert self.size == None,"No need size"
            return {"Lt":Lt.mean()}
        if self.reduction=="max":
            assert self.size ,"size isn't assigned yet"
            lossmax=torch.nn.functional.max_pool1d(Lt.view(1,1,Lt.size()[0])
                                                   ,kernel_size=self.size,
                                                   stride=self.size).squeeze()
            assert lossmax.size()[0]==Lt.size()[0]/self.size,"Check if size is corrected"
            return {"Lt":lossmax.mean()}
    def __call__(self,h0,h1,h2):
        return self.forward(h0,h1,h2)

class Metrics:
    @staticmethod
    def nDCG():
        pass
    @classmethod
    def Avg_nDCG(cls):
        cls.nDCG
        pass
    @staticmethod
    def MPR():
        pass
    @classmethod
    def Avg_MPR(cls):
        cls.MPR
        pass
    @staticmethod
    def AUC(Scores,click,unclick):
        Scores=np.argsort(np.asarray(Scores))[::-1]
        cl_rank=np.asarray([np.where(Scores==cl)[0][0] for cl in click])
        uncl_rank=np.asarray([np.where(Scores==uncl)[0][0] for uncl in unclick])
#        print(cl_rank)
#        print(uncl_rank)
        credit=0
        for cl in cl_rank:
            credit+=np.sum(cl<uncl_rank)
        credit /= (len(cl_rank)*len(uncl_rank))
        return credit
    @classmethod
    def Avg_AUC(cls,ranklist):
        size=len(ranklist)
        totalcredit=0
        for rank in ranklist:
            totalcredit+=cls.AUC(rank["Scores"],rank["Click"],rank["Unclick"])
        return totalcredit/size



if __name__ == "__main__":
    import pickle
    import pandas as pd
    with open("ModelSet.p" ,"rb") as f:
        ModelSet=pickle.load(f)


























