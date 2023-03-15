import math

import torch
import torch.nn as nn
import torch.nn.functional as F
print('Successful import of transformer module')
'''
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
'''

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        #dimension of head
        self.head_dim = embedding_dim/n_heads
        # understand what is n_I
        self.n_heads = n_heads
        self.dropout=dropout

        self.queries = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.keys = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.values = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.r_vec = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.fc_out = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout)

        self.scale = torch.FloatTensor([(embedding_dim) ** 1/2])

    def forward(self, value, key, query, mask=None):
        #Query are the input valie
        batch_size = query.shape[0] 
        Q = self.queries(query)
        K = self.values(value)
        V = self.keys(key)
        R = self.r_vec(query)
        #Current shape of Q,K,V,R = [batch_size, seq_len, embedding_dim]
        #change shape to self tensor 
        queries = self.queries(query).reshape(batch_size, -1, self.num_heads, self.embedding.q_dim)
        keys = self.keys(key).reshape(batch_size, -1, self.num_heads, self.embedding.k_dim)
        value = self.values(value).reshape(batch_size, -1, self.num_heads, self.embedding.v_dim)
        r_vec = self.r_vec(query).reshape(batch_size, -1, self.num_heads, self.embedding.r_dim)

        #permutation to QKV matrix
        Q_permute = queries.permute(0,2,1,3)
        K_permute = key.permute(0,2,1,3)
        V_permute = value.permute(0,2,1,3)
        R_permute = r_vec.permute(0,2,1,3)
        #The numbers in the permute are the dimensions

        # Product between Q and K ==> (Q*K)
        energy = torch.einsum("bhid,bhjd->bhij" , Q_permute ,K_permute)  
        # energy : [batch_size, num_heads, query_position, key_position]

        #if the mask is applied, fills with the -infinity value all the element in the K matrix with a mask==0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Apply the Softmax linear function and dropout 
        attention = self.dropout(F.softmax(energy / self.scale.to(key.device), dim =-1))

        # Final product between attention and V
        final_mul = torch.einsum("bhjd,bhij->bhid", V_permute, attention)
        # output : [batch_size, num_heads, seq_size, V_dimension]

        v_change = (final_mul * R_permute).permute(0,2,1,3)
        # v_change = [batch_size, seq_size, num_heads, v_dim]

        #reshape the self tensor 
        out = v_change.reshape(batch_size, -1, self.num_heads * self.p.v_dim)

        out = self.fc_out(out)

        return out
    


class TransformerBlock (nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention= SelfAttention(embedding_dim,num_heads,dropout)
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
        self.attention = SelfAttention(embedding_dim, num_heads, dropout)
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
    def __init__(self, target_voc_size, embedding_dim, num_heads, num_transformer_block, dropout, device, max_input_len ):
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
            max_length
        )
        self.decoder= Decoder(200, embed_size, heads,num_layers,dropout, device, 200)
        '''self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            dropout,
            device,
            max_length
        )'''

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_masks(self, src, trg):
        # src = [batch_size, src_seq_size]
        # trg = [batch_size, trg_seq_size]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        # trg_mask = [batch_size, 1, trg_seq_size, 1]
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
        torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

        trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)

        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]
        return src_mask.to(src.device), trg_mask.to(src.device)

    def forward(self, src, trg):
        # src = [batch_size, src_seq_size]
        # trg = [batch_size, trg_seq_size]

        src_mask, trg_mask = self.make_masks(src, trg)
        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]

        src = self.embedding(src)
        trg = self.embedding(trg)
        # src = [batch_size, src_seq_size, hid_dim]

        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch_size, src_seq_size, hid_dim]

        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        # out = [batch_size, trg_seq_size, d_x]

        logits = self.embedding.transpose_forward(out)
        # logits = [batch_size, trg_seq_size, d_vocab]

        return logits

def make_src_mask(self, src):
    # src = [batch size, src sent len]
    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(src.device)

def make_trg_mask(self, trg):
    # trg = [batch size, trg sent len]
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(
      torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)

    return trg_mask.to(trg.device)



print('Creating model in module...')
model = Transformer(100, 100, 0, 0)