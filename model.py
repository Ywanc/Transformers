'''
My attempt at replicating the transformer architecture from the "Attention Is All You Need"
with reference from 
https://github.com/hkproj/pytorch-transformer, PyTorch Documentation, ChatGPT
'''
import torch
from torch import nn
from transformers import BertTokenizer
import math

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

d_model = 512
vocab_size = tokenizer.vocab_size

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model, vocab_size): #takes in d_model(embedding dimensions) & vocab_size
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_matrix = nn.Embedding(self.vocab_size, self.d_model)
        
    def forward(self, x): #x should be a LongTensor of ids
        return self.embedding_matrix(x) * math.sqrt(self.d_model)
   
class PositionalEncodings(nn.Module):
    
    def __init__(self, seq_len, d_model, dropout): #takes in seq_len & d_model
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) #randomly zeroes elements of the tensor. to prevent overfitting

        pe_matrix = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        pos = torch.arange(0,seq_len).unsqueeze(1) # (seq_len, 1)                           

        #create tensor of 2i from i=0 to d_model, then /d_model & *ln10000 
        div = torch.exp(torch.arange(0, d_model, 2).float()/d_model * -math.log(10000)) # (d_model//2)

        pe_matrix[:, 0::2] = torch.sin(pos * div) #select all row, col frm 0 with step 2
        pe_matrix[:, 1::2] = torch.cos(pos * div) #select all row, col frm 1 with step 2

        #add batch dimension
        pe_matrix = pe_matrix.unsqueeze(0) #(1, seq_len, d_model). 1 is batch_size

        self.register_buffer("pe_matrix", pe_matrix) #tensor not parameter, but saved in state
    
    def forward(self, x):
        return self.dropout(x + self.pe_matrix[:, :x.shape[1], :]) #only need pe_matrix until length of x

class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h

        # the Query, Key, Value Matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod #independent of instance
    def attention(self, Q, K, V, mask, dropout): #takes in all heads of Q,K,V. (batch, h, seq, d_k), h head of (seq,d_k) matrices
        d_k = Q.shape[-1]
        attention_scores = (Q @ K.transpose(-2,-1))/math.sqrt(d_k) #transpose 1st last & 2nd last dimension. -> (batch, h, d_k, seq)

        if mask is not None:
            # fill -inf in positions where mask==0
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = attention_scores.softmax(dim=-1) #apply softmax across "seq" dimension, -1
        
        if dropout: 
            attention_weights = dropout(attention_weights) 

        return attention_scores @ V

class Transformer(nn.Transformer): # nn.Transformer doesn't have input embedding & pos encoding
    def __init__(self):
        super().__init__()

        

        
        

        

        
    



        


        
        
