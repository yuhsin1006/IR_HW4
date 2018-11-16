
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from tqdm import tqdm

## Open file
fp = open('doc_list.txt','r')
doc_list = [ ]
line = fp.readline()

## 用 while 逐行讀取檔案內容，直至檔案結尾
while line:
    line = line.replace('\n','')
    doc_list.append(line)
    line = fp.readline()
 
fp.close()
print(len(doc_list))

## Open file
fp = open('query_list.txt','r')
query_list = [ ]
line = fp.readline()

## 用 while 逐行讀取檔案內容，直至檔案結尾
while line:
    line = line.replace('\n','')
    query_list.append(line)
    line = fp.readline()
 
fp.close()
print(len(query_list))

voc = [ ]


# In[2]:


# Open file
doc_len = 0
doc_temp = [ ]
doc = [ ]

while doc_len < len(doc_list) :
    fp = open('Document/'+doc_list[doc_len],'r')
    i = 0
    while i < 3:
        buffer = fp.readline()
        i += 1
    
    doc_temp = fp.read()
    doc_temp = doc_temp.replace('-1','')
    doc_temp = doc_temp.split()
    doc = doc + doc_temp
    doc = list(set(doc))

    fp.close()
    doc_len += 1
    
print(len(doc))


# In[3]:


# Open file
query_len = 0
query_temp = [ ]
query = [ ]

while query_len < len(query_list) :
    fp = open('Query/'+query_list[query_len],'r')
    
    query_temp = fp.read()
    query_temp = query_temp.replace('-1','')
    query_temp = query_temp.split()
    query = query + query_temp
    #word = word + query_temp
    query = list(set(query))
    
    fp.close()
    query_len += 1
    
print(len(query))


# In[4]:


voc = doc + query
voc = list(set(voc))
print(len(voc))


# In[5]:


tfd = np.zeros( (len(voc), doc_len), dtype=np.int )
#print(tfd)
tfq = np.zeros( (len(voc), query_len), dtype=np.int )
#print(tfq)


# In[6]:


x = 0
temp = [ ]

while x < doc_len :
    fp = open('Document/'+doc_list[x],'r')
    
    i = 0
    while i < 3:
        buffer = fp.readline()
        i += 1
    
    temp = fp.read()
    temp = temp.replace('-1','')
    temp = temp.split()

    y = 0
    for w in voc :
        for dc in temp :
            if w == dc :
                tfd[y][x] += 1
        y += 1
        
    fp.close()
    x += 1
    
print(tfd)


# In[7]:


#print(np.unique(tfd, axis=0))


# In[8]:


x = 0
while x < query_len :
    fp = open('Query/'+query_list[x],'r')
    
    i = 0
    while i < 3:
        buffer = fp.readline()
        i += 1
    
    temp = fp.read()
    temp = temp.replace('-1','')
    temp = temp.split()

    y = 0
    for w in voc :
        for q in temp :
            if w == q :
                tfq[y][x] += 1
        y += 1
        
    fp.close()
    x += 1
    
print(tfq)


# In[9]:


#print(np.unique(tfq, axis=0))
for i in range(len(voc)) :
    for j in range(doc_len) :
        if tfd[i][j] > 0 :
            tfd[i][j] = math.log(tfd[i][j],2) + 1
            
for i in range(len(voc)) :
    for j in range(query_len) :
        if tfq[i][j] > 0 :
            tfq[i][j] = math.log(tfq[i][j],2) + 1


# In[10]:


idf = np.zeros( (len(voc), 1 ), dtype=np.float )
#print(idf)

for i in range(len(voc)) :
    for j in range(doc_len) :
        if tfd[i][j] > 0 :
            idf[i][0] += 1

print(idf.shape)
#print(np.unique(idf))

for i in range(len(voc)) :
    if idf[i][0] != 0 :
        idf[i][0] = doc_len / idf[i][0]
        idf[i][0] = math.log(idf[i][0],10) 
        
print(idf)


# In[11]:


tfidfd = [ ]
tfidfd = np.zeros( (len(voc), doc_len), dtype=np.float )
print(tfidfd.shape)

tfidfq = [ ]
tfidfq = np.zeros( (len(voc), query_len), dtype=np.float )
print(tfidfq.shape)


# In[12]:


for i in range(len(voc)) :
    for j in range(doc_len) :
        tfidfd[i][j] = tfd[i][j] * idf[i][0]
print(np.unique(tfidfd))        

for i in range(len(voc)) :
    for j in range(query_len) :
        tfidfq[i][j] = tfq[i][j] * idf[i][0]
print(np.unique(tfidfq))


# In[13]:


from sklearn.metrics.pairwise import cosine_similarity

vsm_1 = [ ]
vsm_1 = np.zeros( (query_len, doc_len), dtype=np.float )
print(vsm_1.shape)


# In[ ]:


#vsm_1 = cosine_similarity(tfidfd.T, tfidfq.T)

for i in range(query_len) :
    for j in range(doc_len) :
        vsm_1[i][j] = cosine_similarity([tfidfd[:,j]], [tfidfq[:,i]])

#vsm_1 = vsm_1.T
print(vsm_1.shape)


# In[ ]:


ranking = pd.DataFrame(vsm_1, index = query_list, columns = doc_list)
#print(ranking)


# In[ ]:


ranking = ranking.sort_values(by = query_list[0],ascending= False, axis=1)
ranking


# In[ ]:


Rq = 3

alpha = 0.3
alpha_idf = np.zeros( (len(voc), 1 ), dtype=np.float )

beta = 1
beta_idf = np.zeros( (len(voc), 1 ), dtype=np.float )

new_idf = np.zeros( (len(voc), 1 ), dtype=np.float )

tfidfq_2 = np.zeros( (len(voc), query_len), dtype=np.float )
print(tfidfq_2.shape)

for n in range(query_len) :
    ranking = ranking.sort_values(by = query_list[n],ascending= False, axis=1)
    name = ranking.columns
    name = np.asarray(name)

    #抓出原本的tfidf_doc

    for i in range(Rq) :
        for j in range(doc_len) :
            if name[i] == doc_list[j]:
                #print(name[i], j)
                beta_idf = beta_idf[:,[0]] + tfidfd[:,[j]]
            
    alpha_idf = alpha_idf[:,[0]] + tfidfq[:,[n]]
    alpha_idf = alpha_idf * alpha
    
    beta_idf = beta_idf * beta * (1/Rq)

    new_idf = alpha_idf + beta_idf
    #print(alpha_idf, beta_idf, new_idf)

    tfidfq_2[:,[n]] = new_idf
    #print(tfidfq_2[:,[n]])


# In[ ]:


print(tfidfq_2)


# In[ ]:


print(tfidfd.shape)
print(tfidfq_2.shape)
vsm_2 = [ ]
vsm_2 = np.zeros( (query_len, doc_len), dtype=np.float )
print(vsm_2.shape)


# In[ ]:


#vsm_2 = cosine_similarity(tfidfd.T, tfidfq_2.T)
#print(vsm_2[0][0])
for i in range(query_len) :
    for j in range(doc_len) :
        vsm_2[i][j] = cosine_similarity([tfidfd[:,j]], [tfidfq_2[:,i]])
#         print(vsm_2[i][j])
#         input()

vsm_2 = vsm_2.T
print(vsm_2)
print(vsm_2.shape)


# In[ ]:


ranking = pd.DataFrame(vsm_2, index = query_list, columns = doc_list)
#print(ranking)


# In[ ]:


f = open('M10715086.txt','w') 
f.write("Query,RetrievedDocuments\n")
f.close()

f = open('M10715086.txt','a') 

for i in range(query_len) :
    f.write(query_list[i] + ',')
    ranking = ranking.sort_values(by = query_list[i],ascending= False, axis=1)
    name = ranking.columns
    name = np.asarray(name)
    for j in range(doc_len):
        f.write(name[j] + ' ')
    f.write('\n')

f.close()

