from baseModel import wordEmbedding, decoderLayer, positionEmbedding, clones, half_unet, convBlock, layerNorm, \
    encoderLayer, generator
from torch import nn

import torch


class featureEmebedding(nn.Module):
    def __init__(self, dropout=0.1, hidden_dim=512):
        super(featureEmebedding, self).__init__()
        self.h_unet = half_unet()
        self.fusion_conv = convBlock(input_channel=hidden_dim * 2, output_channel=1, kelnel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(256, 512)
        self.relu = nn.ReLU(inplace=True)
        self.norm = layerNorm(hidden_dim)

    def forward(self, x, bs):
        x = self.h_unet(x)
        x = self.fusion_conv(x)

        d0 = x.size(0)
        x = x.view(bs, d0 // bs, -1)

        return self.norm(self.relu(self.linear(x)))


class encoder(nn.Module):
    def __init__(self, hidden_dim=512, h=8, n=3):
        super(encoder,self).__init__()
        self.encoder_layer = encoderLayer(size=hidden_dim, h=h)
        self.layers = clones(self.encoder_layer, n)
        self.norm = layerNorm(hidden_dim)
        delattr(self, 'encoder_layer')

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class decoder(nn.Module):
    def __init__(self, hidden_dim=512, h=8, n=6):
        super(decoder,self).__init__()
        self.decoder_layer = decoderLayer(size=hidden_dim, h=h)
        self.layers = clones(self.decoder_layer, n)
        self.norm = layerNorm(hidden_dim)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class transformer(nn.Module):
    def __init__(self, tgt_vocab_size,max_len=32):
        super(transformer, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.encoder_embedding = nn.Sequential(
            featureEmebedding(), positionEmbedding(device=self.device, max_len=max_len))
        self.encoder = encoder()
        self.decoder_embedding = nn.Sequential(
            wordEmbedding(vocab=tgt_vocab_size), positionEmbedding(device=self.device, max_len=max_len))
        self.decoder = decoder()
        self.generator = generator(vocab_size=tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        bs = src.size(0)
        len = src.size(1)
        src = src.view(bs*len,-1)
        src = self.encoder_embedding(src, bs)

        out = self.encoder(src, src_mask)

        tgt = self.decoder_embedding(tgt)
        logits = self.decoder(tgt, out, src_mask, tgt_mask)
        return self.generator(logits)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = featureEmebedding()
    model.to(device)
    x = torch.randn(8, 1, 348, 348, device=device)
    out = model(x,2)
    print(out.shape)

