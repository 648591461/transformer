import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器结构
    """
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = source_embed  # 源数据嵌入层
        self.tgt_embed = target_embed  # 目标嵌入层
        self.generator = generator  # 生成器？

    def forward(self, source, target, src_mask, tgt_mask):

        return self.decode(self.encode(source, src_mask), src_mask, target, tgt_mask)

    def encode(self, source, src_mask):
        return self.encoder(source, src_mask)

    def decode(self, source, src_mask, target, tgt_mask):
        return self.decoder(source, src_mask, target, tgt_mask)

class Generator(nn.Module):
    """
    映射到单词表维度并且通过softmax获取每个词的概率
    """

    def __init__(self, input_dim, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(input_dim, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    """
    :param module: 被复制层
    :param N: 复制数量
    :return: 叠加后的模型
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 编码器
class Encoder(nn.Module):
    """
    编码器
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):  # 层标准化
    def  __init__(self, features, eps=1e-6):
        self.a_2 = nn.Parameter(th.ones(features))
        self.b_2 = nn.Parameter(th.zeros(features))
        self.eps = eps  # 避免商为0

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差框架
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 这里对原来最初的形式做了改动
        # 原本应该是：self.norm(x + sublayer(x))
        return x + self.dropout(sublayer(self.norm(x)))


# 编码层
class EncoderLayer(nn.Module):
    """
    编码层
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# 解码层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # 自注意力机制层
        self.src_attn = src_attn  # 编码器加入
        self.feed_forward = feed_forward  # 线性层
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # 框架定义

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[0](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# 掩码
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).satype('unit8')
    return torch.from_numpy(subsequent_mask) == 0


# 注意力
def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    attn = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn = attn.mask_fill(mask == 0, -1e9)

    p_attn = F.softmax(attn, dim=-1)  # 归一化
    if dropout is not None:
        p_attn = dropout(p_attn)
    return th.matmul(p_attn, v), p_attn


# 多头注意力
class MultiHeadAttention(nn.Module):

    def __init__(self, h, dim_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim_model % h == 0

        self.d_k = dim_model // h
        self.h = h
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = q.size(0)

        q, k, v = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]

        x, self.attn = attention(q, k, v, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 前向传播
class PositionwiseFeedForwar(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwar, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len).unsqueee(1)
        div_term = th.exp(th.arange((0, d_model, 2) * - (math.log(10000.0) / d_model)))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
