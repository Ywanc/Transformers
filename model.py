'''
My attempt at replicating the transformer architecture from the "Attention Is All You Need"
with help from the PyTorch Documentation & ChatGPT.
Minimal reference to existing youtube videos. 
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
    
    def __init__(self, seq, d_model): #takes in seq & d_model
        super().__init__()
        self.seq = seq
        self.d_model = d_model
        self.dropout = nn.Dropout()

        pe_matrix = torch.zeros(seq, d_model) # (seq, d_model)
        pos = torch.arange(0,seq).unsqueeze(1) # (seq, 1)                           

        #create tensor of 2i from i=0 to d_model, then /d_model & *ln10000 
        div = torch.exp(torch.arange(0, d_model, 2).float()/d_model * -math.log(10000)) # (d_model//2)

        pe_matrix[:, 0::2] = torch.sin(pos * div) #select all row, col frm 0 with step 2
        pe_matrix[:, 1::2] = torch.cos(pos * div) #select all row, col frm 1 with step 2

        #add batch dimension
        pe_matrix = pe_matrix.unsqueeze(0) #(1, seq, d_model). 1 is batch_size

        self.register_buffer("pe_matrix", pe_matrix) #tensor not parameter, but saved in state
    
    def forward(self, x):
        return pe_matrix + x



        


        
        
