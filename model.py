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
        self.pe_matrix = torch.randn()
        
        pe_matrix = torch.zeros(seq, d_model)
        pos = torch.arange()
    

        
        
        
