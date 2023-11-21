padding_idx = 0
unk = 1
bos_idx = 2
eos_idx = 3
lr = 3e-4

image_size = 178
seq_max_len = 4
image_padding = 0.5
image_bos = 0.
image_eos = 1.

data_folder = './data/P_skeleton'
data_path = './data/new_sentence_mapper.txt'
vocab_path = './tokenizer/new_dict.txt'
with open(vocab_path, 'r', encoding='utf-8') as f:
    tgt_vocab_size = len(f.readlines())

best_model = './run/best_model.pth'
last_model = './run/last_model.pth'
last_info = './run/last.json'

import torch

device = torch.device('cuda')

