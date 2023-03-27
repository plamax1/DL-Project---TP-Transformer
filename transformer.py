import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
print('Successful import of transformer module')

# Source of Inspiration URL:
# https://www.youtube.com/watch?v=U0s0f995w14&t=768s


class SelfAttention(pl.LightningModule):
    # ***### Should be ok, need masks
    def __init__(self, embedding_dim, n_heads, dropout):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim / n_heads
        self.n_heads = n_heads
        self.dropout = dropout

        self.queries = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.keys = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.values = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, value, key, query, mask=None):
        # query = key = value shape: [batch_size, seq_len, hid_dim]
        batch_size = query.shape[0]
        Q = self.queries(query)
        K = self.values(value)
        V = self.keys(key)
        # Shape of Q,K,V = [batch_size, seq_len, embedding_dim]
        head_dim = int(self.embedding_dim / self.n_heads)

        # Reshaping the matrices to make get num_heads head_dim
        Q = Q.reshape(batch_size, -1, self.n_heads, head_dim)
        K = K.reshape(batch_size, -1, self.n_heads, head_dim)
        V = V.reshape(batch_size, -1, self.n_heads, head_dim)
        # Q,K,V shape: [batch_size, seq_len, n_heads, head_dim]

        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        # Product between Q and K ==> (Q*K)
        # energy : [batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e10"))
        attention = self.dropout(
            F.softmax(energy / (self.embedding_dim**(1 / 2)), dim=-1))
        # attention : [batch_size, n_heads, seq_size, seq_size]

        out = torch.einsum("nhql,nlhd->nqhd",
                           [attention,
                            V]).reshape(batch_size,
                                        query.shape[1],
                                        int(self.n_heads * self.head_dim))
        # output : [batch_size, num_heads, seq_size, V_dimension] #WHERE
        # V_dimension is the

        out = self.fc_out(out)
        # Out = [batch_size, seq_size, d_x]
        return out


class TransformerBlock (pl.LightningModule):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, num_heads, dropout)
        self.n1 = nn.LayerNorm(embedding_dim)
        self.n2 = nn.LayerNorm(embedding_dim)
        mul = 4  # This is the forward expansion
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, mul * embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim * mul, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        att_out = self.attention(value, key, query, mask)

        # adding the residual connections
        res_add_out = att_out + query
        # normalization
        out = self.n1(res_add_out)
        out = self.dropout(out)
        forward_out = self.feed_forward(out)

        # adding the residual connections
        out_to_norm = forward_out + out
        out = self.n2(out_to_norm)
        out = self.dropout(out)
        return out


class Encoder (pl.LightningModule):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            num_trans_block,
            num_heads,
            dropout,
            max_input_len):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_input_len, embedding_dim)
        # define all the layers for all the transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_trans_block)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask):

        N, seq_lenght = input.shape
        tok_embedding = self.token_embedding(input)
        # Embedding shape [batch_size, seq_len, embedding_size]
        pos = torch.arange(0, seq_lenght).expand(N, seq_lenght).to(self.device)
        pos_embedding = self.positional_embedding(pos)
        # Sum the position encoding + token encoding to get the positional
        # encoding
        pos_encoded = tok_embedding + pos_embedding
        out = self.dropout(pos_encoded)
        for i in self.transformer_blocks:
            out = i(out, out, out, mask)
        return out


class DecoderBlock(pl.LightningModule):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, num_heads, dropout)
        self.n1 = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(
            embedding_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, value, key, src_mask, trg_mask):
        attention_out = self.attention(input, input, input, trg_mask)
        # adding residual connections
        add_res = attention_out + input
        out = self.n1(add_res)
        out = self.dropout(out)
        # now it comes encoder attention
        out = self.transformer_block(value, key, out, src_mask)
        return out


class Decoder (pl.LightningModule):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            num_heads,
            num_transformer_block,
            dropout,
            max_input_len):
        super(Decoder, self). __init__()
        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_input_len, embedding_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, dropout)
             for _ in range(num_transformer_block)]
        )

        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, enc_out, src_mask, trg_mask):
        N, seq_len = input.shape
        embedding = self.tok_embedding(input)
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        # Sum the position encoding + the token encoding to get the positional
        # encoding
        pos_encoded = embedding + self.positional_embedding(pos)

        out = self.dropout(pos_encoded)
        for i in self.decoder_blocks:
            out = i(pos_encoded, enc_out, enc_out, src_mask, trg_mask)
            # DecoderBlock(self, input, value, key, src_mask, trg_mask):

            out = self.linear(out)
            # shape[batch_size, seq_len, vocab_size]
            prob_out = self.softmax(out)
        return prob_out


class Transformer(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_size=512,
        num_layers=6,
        heads=8,
        dropout=0,
        max_input_len=200
    ):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_size,
            embed_size,
            num_layers,
            heads,
            dropout,
            max_input_len)
        self.decoder = Decoder(
            vocab_size,
            embed_size,
            heads,
            num_layers,
            dropout,
            max_input_len)
        self.pad_idx = 0

    def make_src_mask(self, src):
        # src = [batch size, src sent len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(src.device)

    def make_trg_mask(self, trg):
        # trg = [batch size, trg sent len]
        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones(
                (trg_len,
                 trg_len),
                dtype=torch.uint8,
                device=trg.device))

        trg_mask = trg_pad_mask.type(
            torch.ByteTensor) & trg_sub_mask.type(
            torch.ByteTensor)

        return trg_mask.to(trg.device)

    def forward(self, src, trg):
        # src = [batch_size, src_seq_size]
        # trg = [batch_size, trg_seq_size]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch_size, 1, 1, pad_seq]
        # trg_mask = [batch_size, 1, pad_seq, past_seq]
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, betas=(
                0.9, 0.995), eps=1e-9)

        return optimizer

    def crossEntropyLoss(self, logits, labels):
        return nn.CrossEntropyLoss(ignore_index=0)(
            logits.reshape(-1, 73), torch.flatten(labels))

    def training_step(self, train_batch):
        x = train_batch[0]
        y = train_batch[1]
        logits = self.forward(x, y)
        loss = self.crossEntropyLoss(logits, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, other):
        x = test_batch[0]
        y = test_batch[1]
        x = x.view(x.size(0), -1)
        logits = self.forward(x, y)
        loss = self.crossEntropyLoss(logits, y)
        prediction = torch.argmax(logits, dim=-1)
        accuracy = torch.sum(y == prediction).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        self.log_dict(output)
        return output
