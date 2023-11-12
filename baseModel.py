import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import copy

c = copy.deepcopy

def clones(module, n):
    return nn.ModuleList([c(module) for _ in range(n)])

class convBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, padding=0,kelnel_size = 3):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kelnel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class downsampleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, need_downsample=True, downsample_type='pooling'):
        super(downsampleBlock, self).__init__()
        self.conv_seq = nn.Sequential(
            convBlock(input_channel=input_channel, output_channel=output_channel),
            convBlock(input_channel=output_channel, output_channel=output_channel),
        )

        self.need_downsample = need_downsample

        if need_downsample:
            if downsample_type == 'pooling':
                self.downsample_layer = nn.MaxPool2d(kernel_size=2)
            else:
                self.downsample_layer = convBlock(input_channel=output_channel, output_channel=output_channel, stride=2,
                                                  padding=1)

    def forward(self, x):
        if self.need_downsample:
            x = self.downsample_layer(x)
        x = self.conv_seq(x)
        return x


class half_unet(nn.Module):
    def __init__(self, initial_channel=1, downsample_layers=4, initial_kernel_num=64):
        super(half_unet, self).__init__()

        self.first_downsample = downsampleBlock(input_channel=initial_channel, output_channel=initial_kernel_num,
                                                need_downsample=False)

        self.latter_stages = nn.ModuleList([
            downsampleBlock(input_channel=initial_kernel_num * pow(2, i - 1),
                            output_channel=initial_kernel_num * pow(2, i))
            for i in range(1, downsample_layers)
        ])

        self.finally_channel = initial_kernel_num * pow(2, downsample_layers)
        self.finally_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            convBlock(input_channel=self.finally_channel//2,output_channel=self.finally_channel)
        )

    def forward(self, x):
        x = self.first_downsample(x)

        for downsample_layer in self.latter_stages:
            x = downsample_layer(x)
        return self.finally_conv(x)


class positionEmbedding(nn.Module):
    def __init__(self, hidden_dim = 512, dropout=0.1, device=None, max_len=32):
        super(positionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim, device=device)

        position = torch.arange(0., max_len, device=device).unsqueeze(1)

        div_term = torch.exp(torch.arange(0., hidden_dim, 2, device=device) * -(math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class multiHeadAttention(nn.Module):
    def __init__(self, h, hidden_dim = 512, dropout=0.1):
        super(multiHeadAttention, self).__init__()

        assert hidden_dim % h == 0

        self.d_k = hidden_dim // h

        self.h = h

        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class layerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(layerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2

class sublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(sublayerConnection, self).__init__()
        self.norm = layerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

class wordEmbedding(nn.Module):
    def __init__(self, vocab,hidden_size = 512):
        super(wordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):

        return self.lut(x) * math.sqrt(self.hidden_size)

class positionWiseFeedForward(nn.Module):
    def __init__(self,d_ff = 1024, hidden_dim = 512 , dropout=0.1):
        super(positionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_dim , d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_dim )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class encoderLayer(nn.Module):
    def __init__(self, size = 512, h=8, dropout =0.1):
        super(encoderLayer, self).__init__()
        # hidden_dim
        self.size = size

        self.self_attn = multiHeadAttention(h=h,hidden_dim=size)
        self.feed_forward = positionWiseFeedForward(d_ff=size*2,hidden_dim=size)
        self.sublayer = clones(sublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class decoderLayer(nn.Module):
    def __init__(self, h=8, dropout =0.1,size=512):
        super(decoderLayer, self).__init__()
        self.size = size
        self.attn = multiHeadAttention(h=h,hidden_dim=size)
        self.self_attn = c(self.attn)
        self.src_attn = c(self.attn)
        self.feed_forward = positionWiseFeedForward(d_ff=size*2,hidden_dim=size)
        self.sublayer = clones(sublayerConnection(size, dropout), 3)
        delattr(self, 'attn')

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        return self.sublayer[2](x, self.feed_forward)

class generator(nn.Module):
    def __init__(self, vocab_size,hidden_dim=512):
        super(generator, self).__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)