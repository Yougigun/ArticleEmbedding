
# coding: utf-8

# In[1]:


import torch


# In[204]:


class Encoder(torch.nn.Module):
    def __init__(self,in_features,embedding_features):
        super(Encoder,self).__init__()
        self.linear1=torch.nn.Linear(in_features,2000)
        self.linear2=torch.nn.Linear(2000,embedding_features)
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
        self.p=torch.rand(in_features)
        self.register_buffer('p_const', self.p)
    def forward(self,x):
        if self.training:
            x=x*(1-torch.rand_like(x,requires_grad=False))
        else:
            x=x*(1-self.p_const)
        ##Layer1
        x=self.linear1(x)
        x=self.tanh(x)
        ##Layer2
        x=self.linear2(x)
        x=self.sigmoid(x)
        ## Calibration
        fb=self.sigmoid(self.linear2.bias)
        x=x-fb
        
        return x
       


# In[214]:


class Decoder(torch.nn.Module):
    def __init__(self,embedding_features,in_features):
        super(Decoder,self).__init__()
        self.linear1=torch.nn.Linear(embedding_features,2000)
        self.linear2=torch.nn.Linear(2000,in_features)
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
    def forward(self,x):
        ##linear 1 
        x=self.linear1(x)
        x=self.tanh(x)
        ##linear 2
        x=self.linear2(x)
        x=self.sigmoid(x)
        return x


# In[ ]:


class EncoderOneLayer(torch.nn.Module):
    def __init__(self,in_features,embedding_features):
        super(EncoderOneLayer,self).__init__()
        self.linear1=torch.nn.Linear(in_features,embedding_features)
        self.sigmoid=torch.nn.Sigmoid()
        self.p=torch.rand(in_features)
        self.register_buffer('p_const', self.p)
    def forward(self,x):
        if self.training:
            x=x*(1-torch.rand_like(x,requires_grad=False))
        else:
            x=x*(1-self.p_const)
        ##Layer1
        x=self.linear1(x)
        x=self.sigmoid(x)
        ## Calibration
        fb=self.sigmoid(self.linear1.bias)
        x=x-fb
        
        return x


# In[ ]:


class DecoderOneLayer(torch.nn.Module):
    def __init__(self,embedding_features,in_features):
        super(DecoderOneLayer,self).__init__()
        self.linear1=torch.nn.Linear(embedding_features,in_features)
#         self.linear2=torch.nn.Linear(2000,in_features)
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
    def forward(self,x):
        ##linear 1 
        x=self.linear1(x)
#         x=self.tanh(x)
        ##linear 2
#         x=self.linear2(x)
        x=self.sigmoid(x)
        return x

