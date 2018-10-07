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
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs 
import torch.nn.functional as F
import torch.utils.data as D
from torch.autograd import Variable

torch.manual_seed(1)

class BiLSTM_ATT(nn.Module):
    def __init__(self,vocab_size,tag_size,embedding_dim,hidden_dim,pos_size,pos_dim,batch):
        super(BiLSTM_ATT,self).__init__()
        self.batch = batch
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        self.pos_size = pos_size
        self.pos_dim = pos_dim
        
        self.word_embeds = nn.Embedding(self.vocab_size,self.embedding_dim)
        #self.word_embeds.weight = nn.Parameter(torch.FloatTensor(weight))
        
        self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size,self.hidden_dim)
        
        self.gru = nn.GRU(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True)
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.tag_size)
        
        self.dropout_emb=nn.Dropout(p=0.3)
        self.dropout_gru=nn.Dropout(p=0.3)
        self.dropout_att=nn.Dropout(p=0.5)
        
        self.hidden = self.init_hidden()
        
        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch,self.tag_size,1))
        
    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2),
                torch.randn(2, self.batch, self.hidden_dim // 2))
                
    def attention(self,H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight,M),2)
        a = torch.transpose(a,1,2)
        return torch.bmm(H,a)
        
    
                
    def forward(self,sentence,pos1,pos2):

        #self.hidden = self.init_hidden()
        self.hidden = self.init_hidden_lstm()
        #print sentence.size(),self.word_embeds(sentence).size()
        embeds = torch.cat((self.word_embeds(sentence),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),2)
        
        embeds = torch.transpose(embeds,0,1)
        #embeds = self.dropout_emb(embeds)
        #gru_out,self.hidden = self.gru(embeds,self.hidden)
        gru_out, self.hidden = self.lstm(embeds, self.hidden)#其实是lstm
        
        gru_out = torch.transpose(gru_out,0,1)
        gru_out = torch.transpose(gru_out,1,2)
        
        gru_out = self.dropout_gru(gru_out)
        att_out = F.tanh(self.attention(gru_out))
        #att_out = self.dropout_att(att_out)
        
        relation = torch.tensor([i for i in range(self.tag_size)],dtype = torch.long).repeat(self.batch, 1)

        relation = self.relation_embeds(relation)
        
        res = torch.add(torch.bmm(relation,att_out),self.relation_bias)
        
        res = F.softmax(res,1)
        #res= torch.transpose(res,1,2)
        
        return res.view(self.batch,-1)
        
        
        
        
        
        
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
TAG_SIZE = len(relation2id)
BATCH = 16
EPOCHS = 50

POS_SIZE = 82
POS_DIM = 25


model = BiLSTM_ATT(len(word2id)+1,TAG_SIZE,EMBEDDING_DIM,HIDDEN_DIM,POS_SIZE,POS_DIM,BATCH)
#model = torch.load('data/model_2000c2.pkl')
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
    acc=0
    total=0
    print "train len",len(train)
    for sentence,pos1,pos2,tag in train_dataloader:
        #model.zero_grad()
        
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        #print sentence.size()
        y = model(sentence,pos1,pos2)
        
        tags = Variable(tag)
        #print y
        loss = criterion(y, tags)
        #print loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # 更新参数
       
        y = np.argmax(y.data.numpy(),axis=1)
        #print y
        #print tag
        for y1,y2 in zip(y,tag):
            if y1==y2:
                acc+=1
            total+=1
        if total%6000==0:
            print total,100*float(acc)/total,"%"
            print y
            print tag.data.numpy().tolist()
            print loss            
    
    acc_t=0
    total_t=0
    for sentence,pos1,pos2,tag in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)
        y = np.argmax(y.data.numpy(),axis=1)
        for y1,y2 in zip(y,tag):
            if y1==y2:
                acc_t+=1
            total_t+=1
    if total_t!= 0:    
        print "test:",100*float(acc_t)/total_t
    else:
        print "test:0%"
    #if epoch%10==0:
    #    st = "data/model_epoch"+str(epoch)+".pkl"
    #    torch.save(model, st)


torch.save(model, "./model/model_01.pkl")
print "model2 has been saved"


