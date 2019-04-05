
# coding: utf-8

# In[ ]:


import pandas as pd
import scipy.stats as s
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


rdata=pd.read_csv("data.csv")


# In[ ]:


mdata=rdata[rdata['diagnosis']=='M']
mdata.shape


# In[ ]:


bdata=rdata[rdata['diagnosis']=='B']


# In[ ]:


mtrain=mdata.iloc[0:200,2:4]
mtest=mdata.iloc[200:,2:4]
mtest.shape


# In[ ]:


btrain=bdata.iloc[0:200,2:4]
btest=bdata.iloc[200:,2:4]
btrain


# In[ ]:


plt.scatter(btrain['texture_mean'],btrain['radius_mean'])
plt.show()


# In[ ]:


plt.scatter(mtrain['texture_mean'],mtrain['radius_mean'])
plt.show()


# In[ ]:


scapm=mtrain.cov()
scapm=np.array(scapm)


# In[ ]:


mcapm=mtrain.mean()
mcapm=np.array(mcapm)
btest


# In[ ]:


scapm.shape


# In[ ]:


mcapm.shape


# In[ ]:


mcapm.reshape(2,1)


# In[ ]:


scapb=btrain.cov()
scapb=np.array(scapb)


# In[ ]:


mcapb=btrain.mean()
mcapb=np.array(mcapb)


# In[ ]:


mcapb


# In[ ]:


scapb


# In[ ]:


scapm


# In[ ]:


pcapb=btrain.shape[0]/400
pcapm=mtrain.shape[0]/400
pcapb


# In[ ]:


def multivariatenaivebayesclassifier(oneexample):
    #(p(stest(radius)/benign))=normal distribution pdf (gaussian formulae)
    pstesttumorisbenign=s.multivariate_normal.pdf(oneexample,mcapb,scapb)
    #prior probability(P(A))=(p(benign))=pcapb
    num=pstesttumorisbenign*pcapb
    #P(~A)=(p(stest(radius)/malignant))
    pstesttumorismalignant=s.multivariate_normal.pdf(oneexample,mcapm,scapm)
    #p(B)=p(benign)+p(malignant)
    den2=pstesttumorismalignant*pcapm
    pfinalbenign=num/(num+den2)
    return pfinalbenign


# In[ ]:


btest.iloc[:,0:2]


# In[ ]:


truepositivecount=0
falsepositivecount=0
truenegativecount=0
falsenegativecount=0

#benign- negative
#malignant- positive
#format-(right/wrong,benign(-)/malignant(+)count....bta rha h)
for i in range(0,len(btest)):
    pisbenign=multivariatenaivebayesclassifier(btest.iloc[i,0:2])
    if pisbenign > 0.5:
        truenegativecount+=1#benign tha or benign hi bta rha h
    else:
        falsepositivecount+=1#benign tha, galat bta rha h isliye false, and tumor malignant bta rha h
print(truenegativecount)   
print(falsepositivecount)


# In[ ]:


for i in range(0,len(mtest)):
    pisbenign=multivariatenaivebayesclassifier(mtest.iloc[i,:])
    if pisbenign < 0.5:
        truepositivecount+=1#benign tha or benign hi bta rha h
    else:
        falsenegativecount+=1#benign tha, galat bta rha h isliye false, and tumor malignant bta rha h
print(truepositivecount) 
print(falsenegativecount)


# In[ ]:


precision=(truenegativecount+truepositivecount)/(btest.shape[0]+mtest.shape[0])*100
precision


# In[ ]:


recall=(truepositivecount/(mtest.shape[0]))
recall

