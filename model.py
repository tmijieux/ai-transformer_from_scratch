from dataclasses import dataclass
import torch
import torch.nn as nn
import math


@dataclass
class InputParams:
    vocab_size: int
    seq_len: int


@dataclass
class Params:
    nb_layers: int = 6
    nb_heads: int = 8 # each head of size 64
    embed_size: int = 512
    feed_forward_size: int = 2048
    dropout: float = 0.1



class InputEmbedding(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.mult_factor = math.sqrt(embed_size)

    def forward(self, x):
        return self.mult_factor * self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int,  seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positional_encodings = torch.zeros(seq_len, embed_size)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)# (seq_len,1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        positional_encodings = positional_encodings.unsqueeze(0)

        #self.positional_encodings = positional_encodings

        self.register_buffer('pe', positional_encodings)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        # x : (batch, seq_len, embed_size)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, p: Params):
        super().__init__()

        self.linear_1 = nn.Linear(p.embed_size, p.feed_forward_size) # W1 and b1
        self.dropout = nn.Dropout(p.dropout)
        self.linear_2 = nn.Linear(p.feed_forward_size, p.embed_size) # W2 and b2

    def forward(self, x):
        # (x :batch of embedded sentence)  (batch, seq_len, embed_size)
        # --linear1--> (batch, seq_len, d_ff)
        # --linear2--> (batch, seq_len, embed_size)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, p: Params):
        super().__init__()

        assert p.embed_size % p.nb_heads == 0, "embed is not divisible by nb_head"
        self.d_k = p.embed_size // p.nb_heads
        self.nb_heads = p.nb_heads

        D = p.embed_size
        self.w_q = nn.Linear(D, D) # weight query
        self.w_k = nn.Linear(D, D) # weight key
        self.w_v = nn.Linear(D, D) # weight value
        self.w_o = nn.Linear(D, D) # weight output
        self.dropout = nn.Dropout(p.dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]


        #  query is b, h, seq_len, d_k
        #  key is b, h, seq_len, d_k
        # -> key^T is b, h, d_k, seq_len
        # query @ key^T = b, h, seq_len, seq_len
        # basically for each batch, for each head,
        # we have a square matrix seq_len matrix
        # at i,j we have scalar product of word i and j (for the embedding direction concerned by head)

        # (Batch, h, Seq_len, d_k) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # "block scalar product" with same head of other words?
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # softmax on each row
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, embed_size) --> (batch, seq_len, embed_size)
        __key = self.w_k(k) # (batch, seq_len, embed_size) --> (batch, seq_len, embed_size)
        value = self.w_v(v) # (batch, seq_len, embed_size) --> (batch, seq_len, embed_size)

        # (batch, seq_len, embed_size) --> (batch, seq_len, h, d_k) -->(transpose) (batch, h, seq_len, h, d_k)
        d_k = self.d_k
        h = self.nb_heads
        query = query.view (query.shape[0], query.shape[1], h, d_k).transpose(1, 2)
        __key = __key.view (__key.shape[0], __key.shape[1], h, d_k).transpose(1, 2)
        value = value.view (value.shape[0], value.shape[1], h, d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, __key, value, mask, self.dropout)

        # (batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, embed_size)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, h * d_k)

        # (Batch, Seq_Len, embed_size) -> (Batch, Seq_Len, embed_size)
        return self.w_o(x)

class AddNorm(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, p: Params):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(p)
        self.feed_forward = FeedForwardBlock(p)
        self.addNorm = nn.ModuleList([AddNorm(p.dropout) for _ in range(2) ])

    def forward(self, x, src_mask):
        x = self.addNorm[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.addNorm[1](x, self.feed_forward)

        return x


class Encoder(nn.Module):
    def __init__(self, p: Params):
        super().__init__()

        layers = []
        for _ in range(p.nb_layers):
            layers.append(EncoderBlock(p))

        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, p: Params):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(p)
        self.cross_attention = MultiHeadAttentionBlock(p)
        self.feed_forward = FeedForwardBlock(p)

        self.addNorm = nn.ModuleList([AddNorm(p.dropout) for _ in range(3) ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.addNorm[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.addNorm[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.addNorm[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    def __init__(self, p: Params):
        super().__init__()

        layers = []
        for _ in range(p.nb_layers):
            layers.append(DecoderBlock(p))

        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int):
        super().__init__()

        # project embedding vector (i.e vector that store "features"/ideas about words)
        # to a array of probability of each word
        # index of output array is token id
        # value at index i is probability of token i

        self.proj = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, embed_size) -- > (Batch, Seq_Len, vocab_size)

        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
            self,
            p: Params,
            src: InputParams,
            tgt: InputParams,
    ):
        super().__init__()
        self.params =  p
        self.src_params = src
        self.tgt_params = tgt

        self.src_embedding = InputEmbedding(p.embed_size, src.vocab_size)
        self.src_pos = PositionalEncoding(p.embed_size, src.seq_len, p.dropout)
        self.encoder = Encoder(p)

        self.tgt_embedding = InputEmbedding(p.embed_size, tgt.vocab_size)
        self.tgt_pos = PositionalEncoding(p.embed_size, tgt.seq_len, p.dropout)
        self.decoder = Decoder(p)

        self.projection_layer = ProjectionLayer(p.embed_size, tgt.vocab_size)

        # initialize the parameters for learning faster
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, encoder_mask, tgt, decoder_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, encoder_mask, decoder_mask)

    def project(self, x):
        return self.projection_layer(x)


def get_model(config, vocab_src_len: int, vocab_tgt_len: int):
    model = Transformer(
        Params(embed_size=config["d_model"]),
        src=InputParams(
            vocab_size=vocab_src_len,
            seq_len=config["seq_len"]
        ),
        tgt=InputParams(
            vocab_size=vocab_tgt_len,
            seq_len=config["seq_len"],
        ),
    )
    return model
