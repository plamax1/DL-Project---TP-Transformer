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
        