#coding:utf8
import numpy as np
import pickle
with open('./data/people_relation_train.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)
    train = pickle.load(inp)
    labels = pickle.load(inp)
    position1 = pickle.load(inp)
    position2 = pickle.load(inp)
    

with open('./data/people_relation_test.pkl', 'rb') as inp:
    test = pickle.load(inp)
    labels_t = pickle.load(inp)
    position1_t = pickle.load(inp)
    position2_t = pickle.load(inp)
   
print "train len", len(train)     
print "test len", len(test)   
print "word2id len",len(word2id)

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT
        

        
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
TAG_SIZE = len(relation2id)
BATCH = 128
EPOCHS = 50

POS_SIZE = 82
POS_DIM = 25



model = BiLSTM_ATT(len(word2id)+1,TAG_SIZE,EMBEDDING_DIM,HIDDEN_DIM,POS_SIZE,POS_DIM,BATCH)
#model = torch.load('model/model_c1.pkl')
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(size_average=True)



train = torch.LongTensor(train[:len(train)-len(train)%BATCH])
position1 = torch.LongTensor(position1[:len(train)-len(train)%BATCH])
position2 = torch.LongTensor(position2[:len(train)-len(train)%BATCH])
labels = torch.LongTensor(labels[:len(train)-len(train)%BATCH])
train_datasets = D.TensorDataset(train,position1,position2,labels)
train_dataloader = D.DataLoader(train_datasets,BATCH,True,num_workers=2)


test = torch.LongTensor(test[:len(test)-len(test)%BATCH])
position1_t = torch.LongTensor(position1_t[:len(test)-len(test)%BATCH])
position2_t = torch.LongTensor(position2_t[:len(test)-len(test)%BATCH])
labels_t = torch.LongTensor(labels_t[:len(test)-len(test)%BATCH])
test_datasets = D.TensorDataset(test,position1_t,position2_t,labels_t)
test_dataloader = D.DataLoader(test_datasets,BATCH,True,num_workers=2)


for epoch in range(EPOCHS):
    print "epoch:",epoch
    acc=0
    total=0
    for sentence,pos1,pos2,tag in train_dataloader:
        
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)  
        tags = Variable(tag)
        loss = criterion(y, tags)      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
       
        y = np.argmax(y.data.numpy(),axis=1)

        for y1,y2 in zip(y,tag):
            if y1==y2:
                acc+=1
            total+=1
        
    print "train:",100*float(acc)/total,"%"
      
    acc_t=0
    total_t=0
    count_predict = [0,0,0,0,0,0,0,0,0,0,0,0]
    count_total = [0,0,0,0,0,0,0,0,0,0,0,0]
    count_right = [0,0,0,0,0,0,0,0,0,0,0,0]
    for sentence,pos1,pos2,tag in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)
        y = np.argmax(y.data.numpy(),axis=1)
        for y1,y2 in zip(y,tag):
            count_predict[y1]+=1
            count_total[y2]+=1
            if y1==y2:
                count_right[y1]+=1

    
    precision = [0,0,0,0,0,0,0,0,0,0,0,0]
    recall = [0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(count_predict)):
        precision[i] = float(count_right[i])/count_predict[i]
        recall[i] = float(count_right[i])/count_total[i]
    

    precision = sum(precision)/len(precision)
    recall = sum(recall)/len(recall)    
    print "准确率：",precision
    print "召回率：",recall
    print "f：", (2*precision*recall)/(precision+recall)

    if epoch%10==0:
        model_name = "data/model_epoch"+str(epoch)+".pkl"
        torch.save(model, model_name)
        print model_name,"has been saved"


torch.save(model, "./model/model_01.pkl")
print "model has been saved"


