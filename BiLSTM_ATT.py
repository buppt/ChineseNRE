#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.dropout_emb=nn.Dropout(p=0.5)
        self.dropout_gru=nn.Dropout(p=0.5)
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

        
        return res.view(self.batch,-1)
