#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import re
import os


# In[3]:


data = pd.read_pickle("programs.pkl")


# In[4]:


data.columns = ["id","code","label"]


# In[6]:


data


# In[8]:


label_counts = data.label.value_counts()  


# In[9]:


label_counts


# In[11]:


max(data.label.values)   # label 1-104


# In[46]:


data_num = len(data)
ratio = "3:1:1"
ratios = [int(r) for r in ratio.split(':')]
train_split = int(ratios[0]/sum(ratios)*data_num)
val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
data = data.sample(frac=1, random_state=666)
train = data.iloc[:train_split] 
dev = data.iloc[train_split:val_split] 
test = data.iloc[val_split:] 


# In[54]:


train["partition"] = "train"
dev["partition"] = "dev"
test["partition"] = "test"
all_data = pd.concat([train,dev,test],axis=0)
all_data


# In[55]:


all_data.to_pickle('data.pkl')


# 处理代码数据

# In[263]:


def split_camel(camel_str):
    text=""
    try:
        split_str = re.sub(
            r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+',
            '_',
            camel_str)
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
    return " ".join(split_str.lower().split('_'))

def _clean_data(sent):
    sent = re.sub(r"[^A-Za-z0-9,!?\'\`=<>+-/%*&|]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent=re.sub(r"%","% ",sent)
    sent=re.sub(r"&","& ",sent)
    sent=re.sub(r"="," = ",sent)
    sent=re.sub(r"-"," - ",sent)
    sent=re.sub(r"\*"," * ",sent)
    sent=re.sub(r"/"," / ",sent)
    sent=re.sub(r"<"," < ",sent)
    sent=re.sub(r">","  > ",sent)
    sent=re.sub(r"\|"," | ",sent)
    #sent=re.sub(r"\+"," + ",sent)
    #sent = re.sub(r"\(", " \( ", sent)
    #sent = re.sub(r"\)", " \) ", sent)
    sent = re.sub(r"\?", " ? ", sent)
    
    sent=re.sub(r"\+\+","## ",sent)
    sent=re.sub(r"\+"," + ",sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    sent=re.sub(r'##','++',sent)
    sent=re.sub(r"! =","!=",sent)
    sent=re.sub(r"= =","==",sent)
    sent=re.sub(r"< <","<<",sent)
    sent=re.sub(r"> >",">>",sent)
    sent=re.sub(r"\+ =","+=",sent)
    sent=re.sub(r"- -","--",sent)
    sent=re.sub(r"- =","-=",sent)
    sent=re.sub(r"\* =","*=",sent)
    sent=re.sub(r"/ =","/=",sent)
    sent=re.sub(r'\d+','numicon ',sent)
    sent=re.sub(r"< =","<=",sent)
    sent=re.sub(r"> =",">=",sent)
    #sent=re.sub(r"\[","",sent)
    #sent=re.sub(r"+ +","xy",sent)
    #sent=re.sub(r"\\","",sent)  
    sent=split_camel(sent)
    return sent

def _clean_string(sent):
    sent=re.sub(r"\[","",sent)
    sent=re.sub(r"\]","",sent)
    sent=re.sub(r"\'","",sent)
    return sent


# In[264]:


all_data['code_']=all_data['code'].apply(_clean_data)
for i in range(52001):
    all_data['code_'].iloc[i]=str(all_data['code_'].iloc[i])
all_data['code_']=all_data['code_'].apply(_clean_string)


# In[268]:


dict={}
for index,row in all_data.iterrows():
    for token in row["code_"].split():
        if token in dict:
            dict[token]+=1
        else:
            dict[token]=1 

            
print(len(dict))


# In[269]:


cnt = 1
for k in dict.keys() :
        dict[k] = cnt
        cnt = cnt + 1
cnt


# In[270]:


def turn_id(s):
    return [dict[token] for token in s.split()]

all_data["code_ids"] = all_data["code_"].apply(turn_id)


# In[272]:


train_ = all_data[all_data["partition"]=="train"][["id","code_ids","label"]]
train_.to_csv("train.csv")
dev_ = all_data[all_data["partition"]=="dev"][["id","code_ids","label"]]
dev_.to_csv("dev.csv")
test_ = all_data[all_data["partition"]=="test"][["id","code_ids","label"]]
test_.to_csv("test.csv")

