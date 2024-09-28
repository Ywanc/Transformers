'''
the transformer model using nn.Transformer 
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
    
    def __init__(self, d_model, dropout, max_len=5000): #takes d_model, dropout, max_len computed until 5000 sequences/tokens
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout) #randomly zeroes elements of the tensor. to prevent overfitting

        pe_matrix = torch.zeros(max_len, d_model) # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)                           

        #create tensor of 2i from i=0 to d_model, then /d_model & *ln10000 
        div = torch.exp(torch.arange(0, d_model, 2).float()/d_model * -math.log(10000)) # (d_model//2)

        pe_matrix[:, 0::2] = torch.sin(pos * div) #select all row, col frm 0 with step 2
        pe_matrix[:, 1::2] = torch.cos(pos * div) #select all row, col frm 1 with step 2

        #add batch dimension
        pe_matrix = pe_matrix.unsqueeze(0) #(1, max_len, d_model). 1 is batch_size

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

class Transformer(nn.Transformer):

    def __init__(self, vocab_size, d_model, head, nhid, nlayers, dropout=0.5):
        #initialises nn.Transformer with following parameters
        super().__init__(
            d_model=d_model, nhead=head, dim_feedforward=nhid,
            num_encoder_layers=nlayers, num_decoder_layers=nlayers, dropout=dropout
        )
        self.model_type = 'Transformer' #simple labelling attribute
        self.mask = None # masking
        self.embedding = nn.Embedding(vocab_size, d_model) #create embedding matrix of (vocab_size, d_model)
        self.pos_encoder= PositionalEncodings(d_model ,dropout) # create pe_matrix 
        self.d_model = d_model
        self.init_weights() #custom weight initialisation
        self.projection_layer = nn.Linear(d_model, vocab_size) #projects embedding to vocab_size for softmax

        def init_weights():
            initrange = 0.1
            #uniform initialisation for embedding matrix
            nn.init.uniform_(self.embedding_marix.weight, -initrange, initrange)
            # zero initialisation for projection layer biases
            nn.init.zeros_(self.projection_layer.bias)
            # uniform intialisation for projection layer's weights
            nn.init.uniform_(self.projection_layer.weight, -initrange, initrange)

    def forward(self, src, tgt, src_mask, tgt_mask): # src = seq to encoder, tgt = seq to encoder 
        # src -> [seq_len, batch_size]

        #embedding for encoder & decoder inputs
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        # src, tgt -> [seq_len, batch_size, d_model]

        # add positional_encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        # same shape
        
        #pass src & tgt through the transformer, with masking
        # outputs the output embeddings
        output_embeddings = super().forward(src, tgt, src_mask=src_mask, tgt_mask = tgt_mask)

        
        #(1, )
        output_logits = self.projection_layer(output_embeddings)

        #apply log_softmax across dimension -1 of the output logits
        return F.log_softmax(output_logits, dim=-1)

    # Function to generate the decoder mask to prevent looking at future tokens
    def generate_square_subsequent_mask(self, sz):
        # Generate a mask that prevents attending to future tokens
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

            
            

            




        
        

        

        

        

        
        

        

        
    



        


        
        
