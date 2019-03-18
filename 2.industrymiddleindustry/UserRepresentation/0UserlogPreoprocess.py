#%% Import Package
from datetime import datetime
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from time import sleep
from matplotlib import pyplot as plt
import redis
import json
import pickle
config = json.load(open('./config.json', 'r', encoding='utf8'))
redis_conn=config["redis"]
pool = redis.ConnectionPool(
            host=redis_conn['host'],
            port=redis_conn['port'],
            decode_responses=True,
        )
r=redis.StrictRedis(connection_pool=pool,charset="utf-8")
rediskey=['recsys:item_repz:EncoderState_v0227:item:repz',
 'recsys:item_repz:EncoderState_v0215:item:repz',
 'recsys:item_list:item']
#%% (A)
#%% Read Userlog.csv
userlog=pd.read_csv("./userlog/userlog.csv")
#%% Verify/Change Data type , Save File
userlog["userid"]=userlog["userid"].apply(lambda user : str(user))
userlog["user_pseudo_id"]=userlog["user_pseudo_id"].apply(lambda puser:str(puser))
userlog["uid"]=userlog.apply(lambda r: r["userid"] if r["userid"]!="nan"
                                               else r["user_pseudo_id"],axis=1)
userlog["SelectItem"]=userlog["SelectItem"].apply(lambda SelectItem:str(SelectItem).lower())
userlog["BeforeList"]=userlog["BeforeList"].apply(lambda BeforeList:str(BeforeList).lower())
userlog["AfterList"]=userlog["AfterList"].apply(lambda AfterList:str(AfterList).lower())
userlog["Session_time"]=userlog["event_date"].apply(lambda date : pd.Timestamp(date))
userlog.drop("event_date",inplace=True,axis=1)
userlog.to_pickle("./userlog/userlog.p")

#%% Number of User Sessions
userid=userlog["userid"]
useridlogin=userid["nan"!=userlog["userid"]].value_counts()
useridnologin=userlog["user_pseudo_id"]["nan"==userlog["userid"]]
useridnologin=useridnologin.value_counts()
print("How many unique Userid/Device  : ",len(useridlogin))
print("How many unique User Pseudo ID : ",len(useridnologin))

useridHist=useridlogin.sort_values(ascending=False)
plt.figure("Hist of login user session")
useridHist.plot("hist",bins=100,log=True,title="Hist of login user session")

useridNologinHist=useridnologin.sort_values(ascending=False)
plt.figure("Hist of No login user session")
useridNologinHist.plot("hist",bins=100,log=True,title="Hist of No login user session")
print("How many Sessions by Userid/Device and User Pseudo ID\n",
      (userid!="nan").value_counts().rename({True:"Userid/Device",
                    False:"User Pseudo ID"}).to_frame().rename({"userid":"Sessions"},axis="columns"))
      #.rename({True:"Userid/Device",
                    #False:"User Pseudo ID"},axis="index").rename({"sum":"Sessions",},axis="columns").loc[["Userid/Device","User Pseudo ID"]])

#%% Frequency of Clicked Position in list
FreCliPos=userlog["seq"].groupby(userlog["seq"]).agg(["sum"]).values
print("Minium of Pos :",userlog["seq"].min())
print("Maxium of Pos :",userlog["seq"].max())
plt.figure("Frequency of Clicked Position in list")
plt.title("Frequency of Clicked Position in list")
plt.plot(FreCliPos)
#%% Read UserNewsViewLog.csv (Browsing history)
UserNewsViewLog=pd.read_csv("./userlog/UserNewsViewLog.csv")
#%% UserNewsViewLog Verify/Change Data type , Save File
UserNewsViewLog["Browse_time"]=UserNewsViewLog["event_date"].apply(lambda date : pd.Timestamp(date))
UserNewsViewLog.drop("event_date",inplace=True,axis=1)
UserNewsViewLog["item_id"]=UserNewsViewLog["item_id"].apply(
                                                lambda item : str(item).lower()
                                                            )
UserNewsViewLog["userid"]=UserNewsViewLog["userid"].apply(
                                                        lambda user:str(user)
                                                        )
UserNewsViewLog["user_pseudo_id"]=UserNewsViewLog["user_pseudo_id"].apply(
                                                    lambda puser:str(puser)
                                                    )
UserNewsViewLog["uid"]=UserNewsViewLog.apply(
        lambda r: r["userid"] if r["userid"]!="nan" else r["user_pseudo_id"],axis=1
        )
UserNewsViewLog=UserNewsViewLog.sort_values("Browse_time").reset_index(drop=True)
UserNewsViewLog.to_pickle("./userlog/UserNewsViewLog.p")
#UserNewsViewLog.set_index(["userid","Browse_time"],inplace=True)

#%% Read Userplg.csv (Browsing history)
#userBrosHsty=pd.read_csv("./userlog/LoginUserNewsLog.csv")
##%% userBrosHsty Verify/Change Data type , Save File
#userBrosHsty["Browse_time"]=userBrosHsty["event_date"].apply(lambda date : pd.Timestamp(date))
#userBrosHsty.drop("event_date",inplace=True,axis=1)
#userBrosHsty["item_id"]=userBrosHsty["item_id"].apply(lambda item : str(item).lower())
#userBrosHsty["userid"]=userBrosHsty["userid"].apply(lambda user:str(user))
#userBrosHsty=userBrosHsty.sort_values("Browse_time").reset_index(drop=True)
#userBrosHsty.to_pickle("./userlog/userBrosHsty.p")
##userBrosHsty.set_index(["userid","Browes_time"],inplace=True)
#

#%% Number of user,device in log
_li=[]
A=set(userlog[userlog["userid"]!="nan"]["userid"])
B=set(UserNewsViewLog[UserNewsViewLog["userid"]!="nan"]["userid"])
for i in A:
    if i in B : _li.append(i)
print(len(A))
len(_li)
#%% Number of user pseudo id in log
_li=[]
A=set(userlog[userlog["userid"]=="nan"]["user_pseudo_id"])
B=set(UserNewsViewLog[UserNewsViewLog["userid"]=="nan"]["user_pseudo_id"])
for i in A:
    if i in B : _li.append(i)
print(len(A))
len(_li)
#%% Plot user_pseudo_id and userid browsing times
plt.figure("user_pseudo_id")
UserNewsViewLog[UserNewsViewLog["userid"]=="nan"]["user_pseudo_id"].value_counts().sort_values().plot(logy=True,rot=30)
x=UserNewsViewLog[UserNewsViewLog["userid"]=="nan"]["user_pseudo_id"].value_counts().sort_values()
plt.figure("userid")
UserNewsViewLog[UserNewsViewLog["userid"]!="nan"]["userid"].value_counts().sort_values().plot(logy=True,rot=30)
y=UserNewsViewLog[UserNewsViewLog["userid"]!="nan"]["user_pseudo_id"].value_counts().sort_values()

#%% (B)
#%% Load Data for Combine Sessions and Browsing History
userlog=pd.read_pickle("./userlog/userlog.p")
userlog=userlog.sort_values("Session_time")
userlog=userlog.set_index(["uid","Session_time"])
userlog=userlog.sort_index(0,[0,1],ascending=True)

usernewsviewlog=pd.read_pickle("./userlog/UserNewsViewLog.p")
usernewsviewlog=usernewsviewlog.sort_values("Browse_time")
usernewsviewlog=usernewsviewlog.set_index(["uid","Browse_time"])
usernewsviewlog=usernewsviewlog.sort_index(0,[0,1],ascending=True)

#%% Combine Sessions and Browsing History ,then Save file to usermodeldata.p
Period=14 #Before session, How many days
ModelData=dict()
def combinelist(bef,item,aft):
    if bef=="nan":
        return (item+"|"+aft)[:-1].split("|")
    elif aft=="nan":
        return (bef+item).split("|")
    else :
        return (bef+item+"|"+aft)[:-1].split("|")
for i,((user,SessTime),row) in enumerate(tqdm(userlog.iterrows())):
    UserBrowse=usernewsviewlog.loc[user]
    BrowseTime=UserBrowse.index
    timedelta=pd.Timedelta(days=Period)
    Pick=((SessTime - timedelta)<BrowseTime) & (BrowseTime<SessTime)
    if np.sum(Pick) !=0:
        List=combinelist(row["BeforeList"],row["SelectItem"],row["AfterList"])
        PosInList=List.index(row["SelectItem"])
        if user not in ModelData:
            ModelData[user]=[{"Date":SessTime,"PosInList":row["seq"],
                              "Data":{"Log":UserBrowse[Pick]["item_id"].values,
                                      "Session":{"List":List,
                                                 "Click":set([PosInList]),"Unclick":set()}
                                      }
                              }]
        else:ModelData[user].append({"Date":SessTime,"PosInList":row["seq"],
                              "Data":{"Log":UserBrowse[Pick]["item_id"].values,
                                      "Session":{"List":List,
                                                 "Click":set([PosInList]),"Unclick":set()}
                                      }
                              })
## Save file
pd.to_pickle(ModelData,"./usermodeldata.p")

#%% (C)
#%% Loade usermodeldata
usermodeldata=pd.read_pickle("./usermodeldata.p")
userid=pd.read_pickle("./userid.p")
suserid=userid['session']['userid']
suserpid=userid['session']['userpid']
#%%How many valid session with browsing log
n=0
for u in usermodeldata:
    n+=len(usermodeldata[u])
#%% How many from userid or userpid
n=0
m=0
for u in usermodeldata:
   if u in suserid : n+=len(usermodeldata[u])
   if u in suserpid : m+=len(usermodeldata[u])

#%% (D)
#%% Label ever browsed Item
#%% Load log
usermodeldata=pd.read_pickle("./usermodeldata.p")
usernewsviewlog=pd.read_pickle("./userlog/UserNewsViewLog.p")
usernewsviewlog=usernewsviewlog.sort_values("Browse_time")
usernewsviewlog=usernewsviewlog.set_index(["uid","Browse_time"])
usernewsviewlog=usernewsviewlog.sort_index(0,[0,1],ascending=True)
#%% Label every Click Item in List
users=usermodeldata.keys()
for user in tqdm(users):
    for ses in usermodeldata[user]:
        date=ses["Date"]
        List=np.asarray(ses["Data"]["Session"]["List"])
        logseen=usernewsviewlog.loc[user][:date]["item_id"].values
        mask=np.isin(List,logseen)
        cli=np.where(mask)[0]
        uncli=np.where(mask==False)[0]
        ses["Data"]["Session"]["Click"]=ses["Data"]["Session"]["Click"]|set(cli)
        ses["Data"]["Session"]["Unclick"]=ses["Data"]["Session"]["Unclick"]|set(uncli)

#%% Save usermodeldata.p file
pd.to_pickle(usermodeldata,"./usermodeldata.p")


#%% (E)
#%% Substract non-existent guid in Redis
#%%Load
with open("usermodeldata.p","rb") as f:
    usermodeldata=pickle.load(f)
with open("guidInRedis.p","rb") as f:
    guidInRedis=pickle.load(f)
#%% Substract log not in redis
#guidInRedis=list(guidInRedis)
for user in tqdm(usermodeldata):
    for sess in usermodeldata[user]:
        sess["Data"]["Log"]=[guid for guid in sess["Data"]["Log"]
                                                        if guid in guidInRedis]

        fltr=set([order for order ,guid in enumerate(sess["Data"]["Session"]["List"])
                                                if guid not in guidInRedis ])

        sess["Data"]["Session"]["Click"]=sess["Data"]["Session"]["Click"]-fltr
        sess["Data"]["Session"]["Unclick"]=sess["Data"]["Session"]["Unclick"]-fltr

#%% Substrtact session with no guid in log,cick,Unclick
usermodeldata2=dict()
for user in tqdm(usermodeldata):
    for sess in usermodeldata[user]:
        if len(sess["Data"]["Log"])==0 :break
        if len(sess["Data"]["Session"]["Click"])==0:break
        if len(sess["Data"]["Session"]["Unclick"])==0:break
        if user not in usermodeldata2: usermodeldata2[user]=[sess]
        else : usermodeldata2[user].append(sess)
#%% Save usermodeldata2
with open("usermodeldata2.p","wb") as f:
    pickle.dump(usermodeldata2,f)
#%% How many session
n=0
for user in usermodeldata2:
    n+=len(usermodeldata2[user])
## n=62,472

m=0
for user in usermodeldata:
    m+=len(usermodeldata[user])
## m=68,262

#%% (F)
#%% Preparing Train and Validation set
#%% Load usernodeldata.p
usermodeldata=pd.read_pickle("./usermodeldata2.p")
#%% Split Train set and Validation set
sessions=[]
split=6/7
for user in usermodeldata:
    for sess in usermodeldata[user]:
        sess["User"]=user
for user in usermodeldata:
    sessions+=usermodeldata[user]
randsession=np.random.permutation(sessions)
TrainSet=randsession[:int(np.floor(len(randsession)*split))]
ValiSet=randsession[int(np.floor(len(randsession)*split)):]
##SAVE
pd.to_pickle(usermodeldata,"./usermodeldata.p") ##resave
pd.to_pickle(sessions,"./ModelSet.p")
pd.to_pickle(TrainSet,"./TrainSet.p")
pd.to_pickle(ValiSet,"./ValiSet.p")
#%% EXPORT ALL REQUISITE GUID
guidpool=set()
users=usermodeldata.keys()
for u in tqdm(users):
     for sess in usermodeldata[u]:
         guidpool=guidpool|set(sess["Data"]["Log"])|set(sess["Data"]["Session"]["List"])
#%% Save Guidpool
with open("guidpool.p","wb") as f :
    pickle.dump(guidpool,f)
#%% (G)
#%% Fetch guid reprez in redis
#%% Load guidInRedis and guidpool
with open("guidInRedis.p","rb") as f:
    guidInRedis=pickle.load(f)
with open("guidpool.p","rb") as f:
    guidpool=pickle.load(f)
#%% Fetch repz of guid
guidpool=list(guidpool)
repzs=r.hmget(rediskey[1],guidpool)
guid2repz={guid:repz for guid,repz in zip(guidpool,repzs)}
repzs=r.hmget(rediskey[0],guidpool)

##Filter none repz
for guid,repz in zip(guidpool,repzs):
    if repz != None :
        guid2repz[guid]=repz
##Transform str to number array
for guid in tqdm(guid2repz):
    guid2repz[guid]=np.asarray([float(elm) for elm in guid2repz[guid].split(",")])
#%%
#%% Save guid2repz
with open("guid2repz.p","wb") as f :
    pickle.dump(guid2repz,f)

#%%(H)
#%% Add log2array in ModelSet
#%%
with open("guid2repz.p","rb") as f:
    guid2repz=pickle.load(f)

with open("ModelSet.p","rb") as f:
    ModelSet=pickle.load(f)

#%% Add LogReprz and SessReprz

for sess in ModelSet:
    buffer=[]
    for log in sess["Data"]["Log"]:
        buffer.append(guid2repz[log])
    buffer=np.asarray(buffer)
    sess["Data"]["LogReprz"]=buffer
    buffer=[]
    for g in sess["Data"]["Session"]["List"]:
        buffer.append(guid2repz[g])
    buffer=np.asarray(buffer)
    sess["Data"]["SessReprz"]=buffer
#%% Save
with open("ModelSet.p","wb") as f:
    pickle.dump(ModelSet,f)







































