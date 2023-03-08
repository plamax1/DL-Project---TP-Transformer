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
    def __init__(self, p_emb):
        super(SelfAttention, self).__init__()
        self.p_emb = p_emb
        #dimension of head
        self.head_dim = p_emb.d_x
        #number of heads
        self.n_heads = p_emb.n_heads

        #each h applied separated transformation 
        self.Wq = nn.Linear(self.head_dim, p_emb.d_q * p_emb.n_heads)
        self.Wk = nn.Linear(self.head_dim, p_emb.d_k * p_emb.n_heads)
        self.Wv = nn.Linear(self.head_dim, p_emb.d_v * p_emb.n_heads)
        self.Wr = nn.Linear(self.head_dim, p_emb.d_r * p_emb.n_heads)

        #understand what is n_I
        self.wo = nn.Linear(p_emb.d_v * p_emb.n_heads, p_emb.d_x)

    def forward(self, value, key, query, mask=None):
        #how many times the block is repeated
        N = query.shape[0] 
 
        q = self.Wq(query)
        k = self.Wk(key)
        v = self.Wv(value)
        r = self.Wr(query)

        q = q.view(N, -1, self.num_I, self.p_emb.d_q).permute(0,2,1,3)
        k = k.view(N, -1, self.num_I, self.p_emb.d_k).permute(0,2,1,3)
        v = v.view(N, -1, self.num_I, self.p_emb.d_v).permute(0,2,1,3)
        r = r.view(N, -1, self.num_I, self.p_emb.d_r).permute(0,2,1,3)

        # Product between Q and K ==> (Q*K)
        # matmul_qk -> (N, heads, query_len, key_len) 
        matmul_qk = torch.einsum("nqhd,nkhd->nhqk" , q,k)  
        
        #if the mask is applied, fills with the -infinity value all the element of the matmul_qk with a mask==0
        if mask is not None:
            matmul_qk = matmul_qk.masked_fill(mask == 0, float("-1e20"))
        
        dk= self.embedding_size ** (1/2)

        # Apply the Softmax linear function 
        attention = torch.softmax(matmul_qk / dk, dim =3)

        # Final product between attention and V
        # output -> (N, query_len, heads, head_dim )
        output = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        
        output = self.fully_connect(output)
        return output
    


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
    

class Encoder (nn.Module):
    def __init__(self,vocab_size, embedding_dim, num_trans_block, num_heads, device, dropout, max_input_len):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_input_len, embedding_dim)
        #define all the layers for all the transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim,num_heads, dropout)
                for _ in range(num_trans_block)
            ]
        ) #Cerca se c'è un modo di farlo diverso
        self.dropout = nn.Dropout

    def forward(self, input, mask):
        N, seq_lenght = input.shape #capire che sono e se c'è un altro modo di definirli
        embedding = self.token_embedding(input)
        pos= torch.arange(0, seq_lenght).expand(N, seq_lenght).to(self.device)
        #Sum the position encoding + the token encoding to get the positional encoding
        pos_encoded = embedding + pos
        out = self.dropout(pos_encoded)
        #We compute the output for each transformer block
        for i in self.transformer_blocks:
            out = i(out, out, out, mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.n1 = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(embedding_dim,num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        def forward(self, input, value, key, src_mask, trg_mask):
            attention_out = self.attention(input, input, input, trg_mask)
            #adding residual connection
            add_res= attention_out + input
            out = self.n1(add_res)
            out = self.dropout(out)
            out = self.transformer_block(value, key, out, src_mask)
            return out
        ### Tutta la parte di sopra dell' encoder è definita come transformer block
        ### Capire bene perchè e cosa sono le mask

class Decoder (nn.Module):
    def __init__(self, target_voc_size, embedding_dim, num_heads, num_transformer_block, heads, dropout, device, max_input_len ):
        super(Decoder, self). __init__()
        self.tok_embedding = nn.Embedding(target_voc_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_input_len, embedding_dim)
        self.trasformer_block = nn.ModuleList(
            [DecoderBlock(embedding_dim,num_heads, dropout, device)
             for _ in range (num_transformer_block)]
        )

        self.linear = nn.Linear(embedding_dim, target_voc_size)
        self.dropout= nn.Dropout(dropout)


    def forward(self, input, enc_out, src_mask, trg_mask):
        N, seq_len = input.shape
        embedding = self.token_embedding(input)
        pos= torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        #Sum the position encoding + the token encoding to get the positional encoding
        pos_encoded = embedding + pos
        out = self.dropout(pos_encoded)

        for i in self.trasformer_block:
            out = i(input, enc_out, enc_out, src_mask, trg_mask)
            out = self.linear(out)
    
