import math

import torch
import torch.nn as nn
import torch.nn.functional as F
print('Successful import of transformer module')
import pytorch_lightning as pl


class SelfAttention(pl.LightningModule):
    ###***### Should be ok, need masks
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
        self.fc_out = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, value, key, query, mask=None):
        #Query are the input value
        # query = key = value = [batch_size, seq_len, hid_dim]
        #print('ATTENTION-FORWARD: query shape:', query.shape)
        batch_size = query.shape[0]
        #print('ATTENTION-FORWARD: batch_size:', batch_size)

        Q = self.queries(query)
        K = self.values(value)
        V = self.keys(key)
        #print('ATTENTION-FORWARD: Q: self.queries(query):',Q.shape )
        #Current shape of Q,K,V,R = [batch_size, seq_len, embedding_dim]
        #change shape to self tensor 
        ####################
        ######TEST WITH RESHAPE AND VIEW
        head_dim = int(self.embedding_dim/self.n_heads)
        #print("Head Dim:", head_dim)
        # Reshaping the matrices to make the get head_dim
        #Reshape the embeddinh in n_heads different pieces
        Q = Q.reshape(batch_size, -1, self.n_heads, head_dim)
        K = K.reshape(batch_size, -1, self.n_heads, head_dim)
        V = V.reshape(batch_size, -1, self.n_heads, head_dim)
        #print('QKVR shape: ', Q.shape)

        #permutation to QKV matrix
        #Q_permute = Q.permute(0,2,1,3)
        #K_permute = K.permute(0,2,1,3)
        #V_permute = V.permute(0,2,1,3)
        #The numbers in the permute are the dimensions
        Q_permute = Q
        K_permute = K
        V_permute = V
        #The numbers in the permute are the dimensions
        #print('QKVR shape after (0,2,1,3) permutation: ', Q_permute.shape)
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q,K])
        # Product between Q and K ==> (Q*K)
        #energy = torch.einsum("bhid,bhjd->bhij" , Q_permute ,K_permute)
        
        # energy : [batch_size, num_heads, query_len, key_len]
        # energy : [batch_size, num_heads, seq_len, seq_len]

        #print('Energy shape: ', energy.shape)
        #if the mask is applied, fills with the -infinity value all the element in the K matrix with a mask==0
        if mask is not None:
            #print('MASK SHAPE: ', mask.shape)
            energy = energy.masked_fill(mask == 0, float("-1e10"))
            #print('Mask applied mask...')    
            #print('Masked Energy shape: ', energy.shape)
        # Apply the Softmax linear function and dropout 
        #print('ENERGY SHAPE: ', energy.shape)
        attention = self.dropout(F.softmax(energy / (self.embedding_dim**(1/2)), dim =-1))
        #print('Attention shape: ', energy.shape)
        # attention = [batch_size, n_heads, seq_size, seq_size]

        # Final product between attention and V
        #out = torch.einsum("bhjd,bhij->bhid", V_permute, attention)
        out =  out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(batch_size, query.shape[1], int(self.n_heads*self.head_dim))
        # output : [batch_size, num_heads, seq_size, V_dimension] #WHERE V_dimension is the 
        # dimension of a single attention head
        #print('Final mul shape: ', final_mul.shape)

        out = self.fc_out(out)
        # Out = [batch_size, seq_size, d_x]
        #print('Attention output shape: ', out.shape)
        return out
    


class TransformerBlock (pl.LightningModule):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention= SelfAttention(embedding_dim,num_heads,dropout)

        #defining the two Normalization Layers
        self.n1 = nn.LayerNorm(embedding_dim)
        self.n2 = nn.LayerNorm(embedding_dim)
        #Forward Layer
        mul = 4 #This is the forward expansion
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, mul * embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim* mul, embedding_dim)
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
    

class Encoder (pl.LightningModule):
    def __init__(self,vocab_size, embedding_dim, num_trans_block, num_heads, device, dropout, max_input_len):
        super(Encoder, self).__init__()
        #print('Vocab Size: ', vocab_size )

        #self.device = device

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_input_len, embedding_dim)
        #print('TOK EMBEDDING: ', self.token_embedding)
        #print('POS EMBEDDING: ', self.positional_embedding)
        #define all the layers for all the transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim,num_heads, dropout)
                for _ in range(num_trans_block)
            ]
        ) #Cerca se c'è un modo di farlo diverso
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask):
        #print('Input shape: ' , input.shape)
        N, seq_lenght = input.shape #capire che sono e se c'è un altro modo di definirli
        #print('N: ', N, 'seq_lenght: ', seq_lenght)
        tok_embedding = self.token_embedding(input)
        #print('Token Embedding shape: ', tok_embedding.shape)
        #Embedding shape [batch_size, seq_len(num of token per sentence), embedding_size]
        pos= torch.arange(0, seq_lenght).expand(N, seq_lenght).to(self.device)
        pos_embedding = self.positional_embedding(pos)
        #print('Pos embedding shape: ', pos_embedding.shape)
        #Sum the position encoding + the token encoding to get the positional encoding
        pos_encoded = tok_embedding + pos_embedding
        #print('POS ENCODED ', pos_encoded.shape)
        out = self.dropout(pos_encoded)
        #We compute the output for each transformer block
        for i in self.transformer_blocks:
            out = i(out, out, out, mask)
        return out
    
class DecoderBlock(pl.LightningModule):
    def __init__(self, embedding_dim, num_heads, dropout, device):
        #self.device = device
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, num_heads, dropout)
        self.n1 = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(embedding_dim,num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, value, key, src_mask, trg_mask):
        #print('DECODER BLOCK - input shape', input.shape)
        #print('DECODER BLOCK - value shape', value.shape)
        #print('DECODER BLOCK - key shape', key.shape)
        attention_out = self.attention(input, input, input, trg_mask)
        #print('arrivo qui')
        #print('DECODER BLOCK - attention out shape', attention_out.shape)

        #adding residual connection
        add_res= attention_out + input
        out = self.n1(add_res)
        out = self.dropout(out)
        ### now it comes encoder attention
        out = self.transformer_block(value, key, out, src_mask)
        return out
        ### Tutta la parte di sopra dell' encoder è definita come transformer block
        ### Capire bene perchè e cosa sono le mask

class Decoder (pl.LightningModule):
    def __init__(self, target_voc_size, embedding_dim, num_heads, num_transformer_block, dropout, device, max_input_len ):
        super(Decoder, self). __init__()
        #self.device = device
        self.tok_embedding = nn.Embedding(target_voc_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_input_len, embedding_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embedding_dim,num_heads, dropout, self.device)
             for _ in range (num_transformer_block)]
        )

        self.linear = nn.Linear(embedding_dim, target_voc_size)
        self.dropout= nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, input, enc_out, src_mask, trg_mask):
        #print('Input shape: ' , input.shape)
        N, seq_len = input.shape
        embedding = self.tok_embedding(input)
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        #Sum the position encoding + the token encoding to get the positional encoding
        pos_encoded = embedding + self.positional_embedding(pos)
        #print('DECODER- pos_encoded shape: ', pos_encoded.shape)

        out = self.dropout(pos_encoded)
        #print('DECODER - calling decoder blocks:')
        #print('DECODER - input shape ', input.shape)
        #print('DECODER - enc-out shape ', enc_out.shape)
        for i in self.decoder_blocks:
            out = i(pos_encoded, enc_out, enc_out, src_mask, trg_mask)
            #DecoderBlock(self, input, value, key, src_mask, trg_mask):

            out = self.linear(out) #this has shape[batch_size, seq_len, vocab_size]
            #print('DECODER - before softmax shape ', out.shape)
            prob_out = self.softmax(out)
            #print('DECODER - after_softmax_shape ', prob_out.shape)

        return prob_out
class Transformer(pl.LightningModule):
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

        '''self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            dropout,
            max_length
        )'''
        self.encoder= Encoder(src_vocab_size, embed_size, num_layers, heads, self.device, dropout, 200)
        self.decoder= Decoder(trg_vocab_size, embed_size, heads,num_layers,dropout, self.device, max_input_len=200)
        '''self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            dropout,
            device,
            max_length
        )'''

        #self.src_pad_idx = src_pad_idx
        self.src_pad_idx = 0
        self.trg_pad_idx = trg_pad_idx
        #self.device = device
        self.pad_idx=0


    def make_src_mask(self, src):
        # src = [batch size, src sent len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(src.device)
    
    def make_trg_mask(self, trg):
        # trg = [batch size, trg sent len]
        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
        torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

        trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)

        return trg_mask.to(trg.device)
    
    def forward(self, src, trg):
        # src = [batch_size, src_seq_size]
        # trg = [batch_size, trg_seq_size]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        #print('TRANSFORMER FORWARD mask shapes: src_mask: ', src_mask.shape  )
        #print('TRANSFORMER FORWARD mask shapes: trg_mask: ', trg_mask.shape  )
        
        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]

        #src = self.embedding(src)
        #trg = self.embedding(trg)
        # src = [batch_size, src_seq_size, hid_dim]
        #This encoder takes the input and makes the embedding on its own
        enc_src = self.encoder(src, src_mask)
        #print('ENC_SRC shape: ', enc_src.shape)
        # enc_src = [batch_size, src_seq_size, hid_dim] hid dim should be token embedding dimension

        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        # out = [batch_size, trg_seq_size, d_x]

        #logits = self.embedding.transpose_forward(out)
        # logits = [batch_size, trg_seq_size, d_vocab]

        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        return optimizer
    
    def crossEntropyLoss(self, logits, labels):
        #print('LOSS: logits shape: ', logits.shape)
        #print('LOSS: target shape: ', labels.shape)
        #return nn.NLLLoss()(logits.reshape(-1, 73), torch.flatten(labels).long())
        return nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, 73), torch.flatten(labels))
        
    def training_step(self, train_batch):
        #filelist = ['test.txt']
        #for file in filelist:
         #   print('Loading file : ', file)
          #  train_batch=get_train_iterator(file, 10, voc)
            x = train_batch[0]
            y= train_batch[1]
            logits = self.forward(x, y)
            loss = self.crossEntropyLoss(logits, y)
            self.log('train_loss', loss)
            return loss
    
    def test_step(self, test_batch,):
        x = test_batch[0]
        y= test_batch[1]
        x = x.view(x.size(0), -1)
        logits = self.forward(x)
        loss = self.nllloss(logits, y)
        prediction = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == prediction).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output
