
# coding: utf-8

# In[107]:


import re
import numpy as np
import string
import collections
from collections import Counter 
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords

#stopword list
stop = set(stopwords.words('english'))

#function to preprocess data
def preprocess_docs(docs,dict_flag=0):
    
    cleanr = re.compile('<.*?>')
    str1=' '
    final_string=[]
    vocab =defaultdict(int)
    
    for sent in docs:
        filtered_sentence=[]
        sent=re.sub(cleanr, ' ', sent) 
        sent = re.sub(r'[?|!|\'|"|#]',r'',sent)
        sent = re.sub(r'[.|,|)|(|\|/]',r' ',sent)
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                    if(cleaned_words.lower() not in stop):
                        s=cleaned_words.lower().encode('utf8')
                        filtered_sentence.append(s)
                    else:
                        continue
                else:
                    continue 
        str1 = b" ".join(filtered_sentence) 
        final_string.append(str1.decode("utf-8"))

        
        if(dict_flag==1):
            
            for word in filtered_sentence:
                vocab[word]+=1
            
        
        
    return vocab,final_string
def preprocess_label(label_index,label,mlb=None):
    
    mapped_label=[]
    for l in label:
        mapped_label.append(tuple([label_index[i] for i in l]))
    
    if(mlb!=None):
        Y=mlb.transform(mapped_label)
       
        return mlb,np.array(Y)
    mlb = MultiLabelBinarizer()
    Y=mlb.fit_transform(mapped_label)
    Y_final=[]
    for i in range(Y.shape[0]):
        Y_final.append(np.array(Y[i])/np.float(np.sum(Y[i])))
    return mlb,np.array(Y_final)


#reading training file
f=open('full_train.txt','r')
d=f.readlines()
docs_tr=[]
label_tr=[]
u_label=[]
for line in d:
    temp=line.split('\t')
    label_tr.append([i.strip() for i in temp[0].split(',')])
    u_label.extend(i.strip() for i in temp[0].split(','))
    docs_tr.append(temp[1])
u_label=list(set(u_label))
label_index=dict((u_label[i],i) for i in range(len(u_label)))
#reading test data

f=open('full_test.txt','r')
d=f.readlines()
docs_te=[]
label_te=[]
for line in d:
    temp=line.split('\t')
    label_te.append([i.strip() for i in temp[0].split(',')])
    docs_te.append(temp[1])

X_train,vocab=preprocess_docs(docs_tr,dict_flag=1)
mlb,Y_train=preprocess_label(label_index,label_tr)
X_test,v=preprocess_docs(docs_tr)
mlb,Y_test=preprocess_label(label_index,label_te,mlb)

word_to_index={};
count=1;

for key in vocab:
    if 100<vocab[key]<6000:
        word_to_index[key]=count;
        count+=1;
print len(word_to_index)
for i in range(len(X_train)):
    X_train[i]=[word_to_index[word] for word in X_train[i] if 100<vocab[word]<6000 ]
print len(X_train)

for i in range(len(X_test)):
    X_test[i]=[word_to_index[word] for word in X_test[i] if 100<vocab[word]<6000 ]
print len(X_test)

data_train=np.zeros( ( len(X_train),len(word_to_index) ))
data_test=np.zeros( ( len(X_test),len(word_to_index) ))

for i in range(data_train.shape[0]):
    for key in X_train[i]:
        data_train[i][key-1]=1;

        
for i in range(data_test.shape[0]):
    for key in X_test[i]:
        data_test[i][key-1]=1;

### saving data as a sparse matrix 
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz


save_npz('data_train',coo_matrix(data_train) )
np.save('y_train',Y_train)

save_npz('data_test',coo_matrix(data_test) )
np.save('y_test',Y_test)

