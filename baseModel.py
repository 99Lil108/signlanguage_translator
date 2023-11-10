import torch
from torch import nn


class convBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, padding=0):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
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


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, device=None, max_len=40):
        super(PositionalEncoding, self).__init__()
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
