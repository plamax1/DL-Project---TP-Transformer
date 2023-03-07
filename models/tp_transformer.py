import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionsEncoding(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, dropout, lenght):
        super(PositionsEncoding,self).__init__
        self.dropout = nn.Dropout(dropout)
        self.lenght=lenght
        self.embedding_dim = embedding_dim

        #token encoding
        self.embedding = nn.Embedding(vocab_dim, embedding_dim )
        self.scaling = torch.sqrt(torch.FloatTensor([embedding_dim]))

        #sinusoidal encoding
        pos_enc = torch.zeros(lenght, embedding_dim)
        # create a o matriz on lenght rows, one for each word, 
        #and for each word create embedding_dim cols, for the encoding of each word


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  #size of the queries and key (Q,K)
        self.heads = heads 
        self.head_dim = embed_size // heads

        #assert is used for a debugging mode, return or a true value, otherwise return false with exception between ""
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.head pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        # energy is a product between Q and K ==> (Q*K)
        energy = torch.einsum("nqhd,nkhd->nhqk" , [queries, keys])

        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)    
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim =3)

        # output is a final product between attention piece and values(V)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention_shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # (N, query_len, heads, head_dim )

        out = self.fc_out(out)
        return out
    


class TransformerBlock (nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention= SelfAttention(embedding_dim, num_heads)
        #defining the two Normalization Layers
        self.n1 = nn.LayerNorm(embedding_dim)
        self.n2 = nn.LayerNorm(embedding_dim)
        #Forward Layer
        mul = 4
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, mul * embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, mul * embedding_dim)
        )
        ### Sostituiscilo con 3 moduli separati
        self.dropout=nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        att_out = self.attention(value, key, query, mask)
        # adding the residual connections
        res_add_out = att_out + query
        #normalization
        out = self.n1(res_add_out)
        out = self.dropout(out)
        forward_out = self.feed_forward(out) #cambia in 3 moduli separati
        # adding the residual connections
        out_to_norm=forward_out + out
        out = self.n2(out_to_norm)
        out = self.dropout(out)
        return out


