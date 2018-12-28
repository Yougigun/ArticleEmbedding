
# coding: utf-8

# In[208]:


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
import visdom
vis=visdom.Visdom()
# env="TagBased"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline

#coding:utf-8
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# #有中文出现的情况，需要u'内容


# ### Load W2V associated with tags

# In[2]:


with open("D:dict_tag2vector","rb") as f:
    W2VofTags=pickle.load(f)


# ### Load W2V 

# In[3]:


from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load('D:models/tmp/skpg_vectors.kv', mmap = 'r')


# ## Load Tag

# In[4]:


with open("D:tag2name/tags_tag2name.p","rb") as f:
    tag2name=pickle.load(f)


# In[6]:


name2tag={}
for t in tag2name:
    name2tag[tag2name[t]]=t


# In[82]:


list_industry=["水泥","食品飲料","石化","紡織","電機機械","電器電纜","化學工業",
               "建材居家用品","造紙","鋼鐵金屬","車輛相關","科技相關","營建地產","運輸","觀光休閒娛樂",
               "金融相關","百貨通路","公用事業","控股","生技醫療保健","農林漁牧","航天軍工","能源","傳播出版","綜合",
               "傳產其他","其他","金屬礦採選",
              ]


# In[83]:


dict_industry={"水泥":["水泥","水泥製品","高爐水泥","預拌混凝土"],
               "食品飲料":["大宗物資","方便麵","水產品加工","可可製品","肉品加工","乳製品","油脂","保健食品",
                       "食品加工","茶葉與咖啡相關","農產品加工","製糖","製鹽","餅乾製品","穀類烘焙製品","調味品",
                       "調理食品","磨粉製品","糖果製造","寵物食品","麵粉","麵製品相關","罐頭","食品添加劑"],
               "石化":["ABS","DOP","PA","PE","PP","PS","PVC","SM","加油站","石化業","芳香烴","烯烴","塑膠加工","塑膠皮布","瀝青"],
               "紡織":["AN""CPL","EG""PTA","工業用布","不織布","內衣褲、襪","化纖原料","毛紡","加工絲",
                     "半合成纖維","尼龍加工絲","尼龍粒","尼龍絲","尼龍塔夫塔","尼龍製品","再生纖維","成衣",
                     "成衣製造","成衣銷售／零售","羊毛","羽絨加工","亞克力紗","亞克力棉","芳綸","染整","氨綸",
                     "紡織中游","紡織用聚酯粒","純棉紗","混紡紗","瓶用聚酯粒","魚網","麻紡","粘膠","棉紡","絲織品",
                     "聚丙烯纖維","聚酯加工絲","聚酯紗","聚酯棉","聚酯絲","聚酯纖維","黏扣帶","織布","寶特瓶"],
               "電機機械":"""
               	工具機
工程機械
工業用縫紉機
工業馬達
手工具機
木工機械
包裝機械
污染控制設備
自行車零件
自動化儀器
車用充電相關
空調冷氣設備
軌道運輸設備
重型電氣設備
飛機零件／製造
家用縫紉機
柴油機
海洋工程設備
紡織機械
配電工程
配電盤
動力機械
產業機械
貨櫃製造
通用機械
造船
智能電網
塑膠機械
裝載機
農用機械
電力配送服務
電力設備
電氣自控設備
電氣開關設備
電氣零件與設備
電機
蓄電池相關
儀器、儀表
模具
橡膠機械
機械
機械零組件
機器人
磨料磨具
輸配電設備
龍門機
壓縮機
幫浦
鍋爐
醫療保健設備／裝置
變電設備
變頻器

               """.split(),
               "電器電纜":"""小家電
冷氣機
洗衣機
音響／視聽組合
家電
家電零組件
通訊電纜
照明
電力電纜
電子線
電冰箱
電視機
電線電纜
電器銷售
漆包線
裸銅線
""".split(),
               "化學工業":"""EVA
MDI
PU樹脂
PVC布
TDI
丁二醇
二甲基甲醯胺(DMF)
工業氣體
化學工業
化學產品通路
天然橡膠
正烷屬烴
正極材料
石油支撐劑
石墨
石蠟
合成樹脂
合成橡膠
有機矽
尿素
乳膠相關
其他化工產品
拋光蠟
油封
肥料
保險粉
染料，顏料
氟化物
炸藥
負極材料
氣凝膠
草甘膦
草酸
強酸產品
硫磺
烷基苯
酚
硝化棉
硝酸銀
黃磷
鈦白粉
塗料
溶劑
煤化工
農藥
電池材料相關
電池隔離膜
電解液
碳煙
碳酸鋇
熱可塑橡膠
熱溶膠
膠帶
衛生清潔用品
輪胎
醋酸
橡膠工業
橡膠製品
環氧丙烷
磷化工
磷酸一銨(MAP)
磷酸二銨(DAP)
雙酚A
鹼業
""".split(),
               "建材居家用品":"""	水龍頭
地毯
岩棉
金屬建材
門用金屬
建材
玻璃
家居用品
陶瓷
傢俱
窗簾
釉料
電梯
寢具
輕鋼架
廚具
衛浴設備
複合板
辦公室傢俱
鎖
""".split(),
               "造紙":"""	工業用紙
文化用紙
包裝用紙
家庭用紙
特殊用紙
紙漿
造紙業
""".split(),
               "鋼鐵金屬":"""工具鋼
不鏽鋼
不鏽鋼品
不鏽鋼剪裁加工
不鏽鋼管
不鏽鋼緊固件
不鏽鋼線材
合金鋼
冷軋不鏽鋼
冷軋鋼捲
板鋼
非鐵金屬
型鋼
烤漆鋼捲
特殊鋼
條鋼
釩
棒鋼
稀土
貴金屬
黃金
鈦
鈀
塗鍍鋼捲
鈷
鈾
鉛
鉑
鉭
鉬
銀
銅
熱軋不鏽鋼
熱軋鋼捲
線材、盤元
鋅
鋁
鋁擠型
鋯
鋼板
鋼胚
鋼剪裁加工
鋼筋
鋼結構
鋼管
錫
螺絲、螺帽
鍍鋅鋼捲
鎂合金
鍶
鎢
鎳
鐵礦砂
""".split(),
               "車輛相關":"""	引擎相關
木床／車架
安全氣囊
自行車
自動駕駛車
沙灘車
汽車內裝
汽車沖壓件
汽車服務相關
汽車空調
汽車保險桿
汽車音響
汽車座椅
汽車鈑金
汽車零組件
汽車製造
汽車銷售
汽車整車
汽車融資
汽機車安全零件
車用冷卻系統
車用防盜系統
車用金屬成型
車用玻璃
車用粉末冶金件
車用排氣系統
車用軸承
車用電池
車用鍛件
車用鑄件
車燈
剎車系統
客貨車輛相關
活塞
特種車輛相關
傳動系統
電動車
電動機車
儀表板
輪圈
機車
機車零組件
轉向系統
懸吊系統
""".split(),
               "科技相關":"""3C通路商
3D印表機
3D電視
3D顯示相關
4G通訊設備
5G通訊設備
ASIC
CMOS晶片
CPU
Displayport
DRAM
DRAM記憶體IC
DRAM模組
DSL晶片組
DSP
FLASH記憶體IC
Flash模組
HDMI
IC生產
IC封裝
IC封裝測試
IC基板
IC設計
IC設計軟體
IC測試
IC量測設備
IC零組件通路商
IC製造
IC讀卡機
Internet相關
IO控制IC
IPTV
LCD控制IC
LCD驅動IC
LCD驅動IC封裝
LCD顯示器
LCM
LED
LED封裝
LED設備
LED散熱基板
LED晶粒
LED照明產品
LED磊晶
LED驅動IC
MCU
MEMS
MLCC
OLED
PCB材料
PCB其他設備
PC週邊IC
PC遊戲
POS機系統
PV Inverter
RFID相關
SRAM記憶體IC
SSD控制IC
STB IC
TFT-LCD
Thunderbolt
TN／STN LCD
UPS
USB
二極體
人工智慧
人臉辨識
入口網站
大數據
小型衛星地面站
工業電腦
中小尺吋面板
分離式元件
化合物晶圓
天線
太陽能
太陽能多晶矽
太陽能系統運用
太陽能矽晶圓
太陽能玻璃
太陽能設備
太陽能電池
太陽能電池模組
太陽能導電漿
手寫板
手機
手機外殼
手機按鍵
手機相機模組
手機面板驅動IC
手機產品零售通路
手機晶片相關
手機遊戲
手機零組件
手機製造
手機震動馬達
主機板
功率放大器
半導體材料通路商
半導體設備
平板電腦
生物辨識IC
生物辨識相關
石英元件
企業資源規劃
光阻劑
光通訊
光通訊元件磊晶
光通訊晶片
光罩
光碟片
光碟機／燒錄機
光碟機驅動IC
光學膜
光學鏡片／頭
光纖主動元件
光纖被動元件
光纖設備
光纖零組件
光纖預製棒
印刷電路板
印刷電路板相關
印表機
印表機耗材
安全監控IC
安全監控系統
耳機
行車紀錄器
行動通訊
伺服器
作業系統
低雜訊放大器
低雜訊降頻器
免持聽筒
投影機
投影機零件
系統整合
車用電子
車載影音系統
事務機器
其他IC
其他電子元件代理商
其他電子零件
受話器
抬頭顯示器(HUD)
物聯網裝置
矽晶圓
矽碟機
社群網站
近場通訊(NFC)
金、錫凸塊
非揮發性記憶體
客戶關係管理
封測用設備
封測服務與材料
指紋辨識
玻璃基板
玻璃基板加工
玻纖布
研磨液/墊
穿戴式裝置
紅外線傳輸模組
背光模組
胎壓監測系統
虹膜辨識
面板設備
面板業
面板零組件
音響設備及零件
倒車雷達
倒車影像系統
套裝軟體
射頻前端晶片
射頻前端模組
射頻開關
振盪器
桌上型電腦
氧化物晶圓
消費性IC
消費性電子產品
砷化鎵相關
記憶卡
記憶卡IC
馬達IC
高速傳輸介面IC
乾膜光阻
偏光板
區域網路
商用遊戲機
基地台
專業晶圓代工
彩色濾光片
探針、探針卡
接取設備
掃瞄器
條碼掃描器
液晶
液晶電視
被動元件
被動元件上游
設計IP
設備儀器廠商
軟板
軟板基板
軟體通路／代理
軟體業
通訊服務
通訊設備
通訊設備零組件
連接線材
連接器
麥克風
嵌入式晶片
掌紋辨識
散熱風扇馬達
散熱模組
晶片組
晶片電阻
智慧手錶
智慧手環
智慧卡IC
智慧卡相關
智慧型手機
智慧眼鏡
測試用板卡
無塵室工程
無線充電
無線網路IC
無線網路設備系統(WLAN)
硬碟相關
程式開關
筆記型電腦
筆記型電腦製造
虛擬實境
視訊會議產品
視訊轉換相關
週邊產品
量測儀器
集線器IC
雲端科技
傳輸介面
塑膠膜電容器
微型揚聲器
微型電聲元件
感測元件
搜尋引擎
滑鼠
準系統
資安設備
資料庫
資訊安全
遊戲產業
遊戲機
雷射鑽孔機、鑽頭
電力線載波晶片
電子化工材料
電子支付
電子其他
電子計算機
電子書閱讀器
電子紙
電子商務
電子通路
電子零件元件
電子製造服務
電子錶
電池保護IC
電池相關
電阻
電信／數據服務
電信設備
電容
電晶體
電感
電源供應器
電腦系統業
電腦通路商
電話／傳真機
電聲產品
鉭質電解電容
磁性材料
磁碟陣列控制器
碟片預錄
網路卡
網路卡IC
網路交換器
網路通訊IC
網路電話(VOIP)
語音助理
語音辨識
銅箔
銅箔基板
影音IC
影像感測元件
播放機
數位內容
數位相框
數位相機
數位相機組裝
數位看板
數位電視
數位電視IC
數據機
數據機晶片組
樞紐
模具沖壓
熱敏電阻
熱導管
磊晶
衛星通訊設備
衛星電話
衛星導航
衛星導航晶片組
鋁質電解電容
導電玻璃
導線架
機殼
機殼表面處理
燒錄機
辦公用品設備
儲存設備
壓敏電阻
應用軟體
檢測驗證服務
鎂鋁合金外殼
鍵盤
濾波器
藍寶石基板
藍寶石晶棒
繪圖IC
繪圖卡
類比IC
觸控IC
觸控面板
觸控面板設備
攝影設備
鐵氟龍基板
變壓器
顯示器
顯示器零件
""".split(),
               "營建地產":"""	
工程顧問
水電消防工程
地產
住宅建設
其他營造工程
房屋仲介
物業投資發展
建築設計、營繕裝璜
基礎建設
園林造景
園區開發
墓園經營
幕牆工程
營建
營造工程
環保工程
""".split(),
               "運輸":"""水上運輸
油輪
物流業
空運
客運
倉儲／貨櫃場
高鐵
貨櫃航運
陸運
散裝航運
運輸事業
鐵路運輸服務""".split(),
               "觀光休閒娛樂":"""化妝品
主題樂園
休閒食品零售
休閒娛樂
美容
俱樂部
旅遊
旅館、餐飲
時尚產業
珠寶
陳列展覽相關
博奕相關
渡假山莊
飯店
電影院／劇院
精品
餐飲
鐘錶
""".split(),
               "金融相關":"""
	再保險
投信
投資信託
房地產投資信託
金控
金融其他
金融業
保險經紀
消費金融
租賃
產險
票券
資產管理
壽險
銀行
證券
證金公司

""".split(),
               "百貨通路":"""	
百貨公司
直銷
便利商店
流通業
相片沖印
食品飲料相關通路
家居相關用品通路
書店
連鎖超市
無店舖販售
貿易
郵購
量販店、大賣場
電視購物
藥粧零售相關
""".split(),
               "公用事業":"""公路
天然氣
水利
水資源
水資源設備／耗材
交通號誌
污水處理廠
其他公用事業
海水淡化
能源
停車場
基礎建設營運
港口
路橋
駕訓班
機場
鐵路
""".split(),
               "控股":"控股公司".split(),
               "生技醫療保健":"""	IVD檢驗儀器設備
人造關節
中藥製劑
手術與治療用醫材
牙科植入器材
生技服務
生技特用化學品
生命科學工具
生物科技
生物製劑
生理監測裝置
生理檢測器材
血糖儀
血壓計
西藥製劑
其他手術器械
其他醫療器材
呼吸與麻醉用器具
放射治療設備
物理治療器具
洗腎器具
食品生技
個人保護用器材
原料藥
骨科類器材
動力手術器具
動物用藥
植物工廠
植物新藥
診斷與監測用醫材
新藥研發
農業生技
電動代步車
福祉輔助設備
維生素
輔助與彌補用醫材
檢驗試劑／紙
環保生技
隱形眼鏡
醫用家具
醫用植入材料
醫學影像裝置
醫療服務
醫療耗材
醫療資訊技術
醫療管理服務
醫療器材及服務
醫療器材通路
醫藥流通
醫藥研發外包服務
醫藥產業
體外診斷用醫材
體溫計
""".split(),
               "農林漁牧":"""	水產養殖
林業
初級產業
食用菌種植
家畜，家禽
畜牧業
農產品種植
飼料
漁業
種子生產
""".split(),
               "航天軍工":"""軍火工業
航天軍工
航天通訊零組件
航天機械零組件
航空器、飛機
衛星相關
""".split(),
               "能源":"""	太陽能發電
水力發電
火力發電
石油開採
地熱發電
汽電共生
供熱
油品儲運／分銷
油氣採掘服務
油氣採掘設備與工程
金屬礦採選
頁岩油／氣
風力發電
核能發電
發電設備與零組件
煤
煤炭採選設備
煉油
燃氣儲運／分銷
礦產採掘服務
""".split(),
               "傳播出版":"""	
文化創意產業
出版業
印刷業
有線電視
報紙業
傳播事業
電視台
廣告
影視
""".split(),
               "綜合":"""綜合
""".split(),
               "傳產其他":"""	人力資源
工安產品
工藝品
手提箱
文具
木(竹)加工／製品
打火機
皮革製品
光學
回收、焚化相關
自動販賣機
底片
拉鍊
服務業
玩具
金融機具
金屬加工／製品
保全
耐火材料
消費者服務
烤肉架
高爾夫球桿頭
健身器材
商業服務／顧問
婚宴顧問
教育事業
清潔服務
球具
球場
瓶蓋
眼鏡
眼鏡架
傳產其他
煙草
資產股
運動用品
運動服
運動產業
運動鞋
運動競賽
碳權相關
製帽
製罐
樂器
複合材料
鞋材
鞋業
嬰童用品
殯葬服務
寵物用品
""".split(),
               "其他":["其他"],
               "金屬礦採選":["金屬礦採選"],
               
               
               
               
               
               


               
               
               }
print(dict_industry["傳播出版"])


# ### Save dict_industry 

# In[187]:


# with open("D:dict_industry","wb") as f:
#     pickle.dump(dict_industry,f)


# In[188]:


# with open("D:dict_industry","rb") as f:
#     test=pickle.load(f)


# # New Explore

# In[7]:


with open("D:MergeNews.p","rb") as f:
    News=pickle.load(f)


# In[102]:


News_vip=News[News["from"]==1].reset_index(drop=True)
News_vip.shape


# In[139]:


News_vip["tags_cn"]=News_vip["tags"].progress_apply(lambda x : set([tag2name.get(i) for i in x.split(",")] if x!=None else []))


# # How many TAGS every years

# In[140]:


df=News["tags"].progress_apply(lambda x : len(x.split(","))if x!=None else 0 )


# In[141]:


ax=df.groupby(by=News["publishtime"].dt.year).agg("sum").plot(figsize=(18,6),xticks=range(1983,2019))


# # How many ARTICLES every years

# In[142]:


ax=df.groupby(by=News["publishtime"].dt.year).count().plot(figsize=(18,6),xticks=range(1983,2019))


# # AVERAGE TAGS number of articles

# In[143]:


x=df.groupby(by=News["publishtime"].dt.year).agg("sum")
y=df.groupby(by=News["publishtime"].dt.year).count()


# In[144]:


(x/y).plot()


# In[231]:


dict_=dict(zip(list_industry,np.zeros(len(list_industry))))
# print(dict_)
def classify(x):
    result=set()
    for t in x :
        for i in dict_industry:
            if t in dict_industry[i]:
                result.add(i)
#                 break
    return result
            
df=News_vip["tags_cn"].progress_apply(classify)
# for i in News_vip["tags_cn"]:
#     for j in dict_industry:
#         for ji in j:
#             if ji in i:
                


# In[232]:


News_vip["indusrty_tags"]=df


# In[233]:


News_vip


# # save News vip with tags in industry and tags_cn

# In[234]:


# with open("D:News_vip_with_industrytag","wb") as f:
#     pickle.dump(obj=News_vip,file=f)


# # 計算產業新聞數

# In[252]:


dict_=dict()
for t in df:
    t=",".join(list(t))
    if t not in dict_:dict_[t]=1
    else: dict_[t]+=1


# In[254]:


for i in pd.Series(dict_).index:
    if len(i.split(","))==1:print(i,pd.Series(dict_).loc[i])
##單看單一產業新聞


# In[242]:


df


# ### 將涉及到相關產業的新聞都算進去產業裡面

# In[260]:


dict_=dict(zip(list_industry,np.zeros(len(list_industry),dtype=int)))
for t in tqdm_notebook(df) :
    for i in dict_:
        if i in t :dict_[i]+=1
    
ser=pd.Series(dict_)
ser=ser.sort_values(ascending=False)
ser
# #plot
# ser=pd.Series(dict_)
# ser=ser.sort_values(ascending=False)
# ax=ser.plot(kind='bar',figsize=(12,6),xticks=range(len(ser)),rot=90)
# # ax.set_xticklabels(range(len(ser)))


# ## issus 

# #### 產業分類不明確

# In[121]:


print(News_vip[:1]["guid"])
print(News_vip[:1]["tags_cn"])
## 被標上銀導致產業分類為鋼鐵金屬


# In[128]:


News_vip[:8]["body_token"].values


# In[138]:


len([""])


# #### 有些文章 多個產業
#     ##### 先拿分類明確(只被分到一個產業)的出來訓練
