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
        # understand what is n_I
        self.n_I = p_emb.n_I

        self.Wq = nn.Linear(self.head_dim, p_emb.d_q * p_emb.n_heads)
        self.Wk = nn.Linear(self.head_dim, p_emb.d_k * p_emb.n_heads)
        self.Wv = nn.Linear(self.head_dim, p_emb.d_v * p_emb.n_heads)
        self.Wr = nn.Linear(self.head_dim, p_emb.d_r * p_emb.n_heads)

        self.W_out = nn.Linear(p_emb.d_v * p_emb.n_I, p_emb.d_x)

        self.dropout = nn.Dropout(p_emb.dropout)
        self.dot_scale = torch.FloatTensor([math.sqrt(p_emb.d_k)])
        self.mul_scale = torch.FloatTensor([1./math.sqrt(math.sqrt(2)-1)])

    def forward(self, value, key, query, mask=None):
        #how many times the block is repeated
        batch_size = query.shape[0] 
 
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)
        R = self.Wr(query)

        Q = Q.view(batch_size, -1, self.num_I, self.p_emb.d_q).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.num_I, self.p_emb.d_k).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.num_I, self.p_emb.d_v).permute(0,2,1,3)
        R = R.view(batch_size, -1, self.num_I, self.p_emb.d_r).permute(0,2,1,3)

        # Product between Q and K ==> (Q*K)
        # matmul_qk -> (batc_size, num_heads, query_position, key_position) 
        matmul_qk = torch.einsum("bhid,bhjd->bhij" , Q,K)  
        
        #apply the dot scale to QK multiplication
        dot = matmul_qk / self.dot_scale.to(key.device)

        #if the mask is applied, fills with the -infinity value all the element in the K matrix with a mask==0
        if mask is not None:
            dot = dot.masked_fill(mask == 0, float("-1e10"))
        
        # Apply the Softmax linear function 
        attention = self.dropout(F.softmax(dot, dim =-1))

        # Final product between attention and V
        # output -> (batch_size, num_heads, seq_size, V_dimension )
        dot_v = torch.einsum("bhjd,bhij->bhid", V, attention)
        
        #capire perche' ha creato questa nuova V
        new_v = dot_v * R
        new_v = new_v.permute(0,2,1,3).contiguous()

        new_v = new_v.view(batch_size, -1, self.n_I * self.p.d_v)

        output = self.W_out(new_v)

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
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
