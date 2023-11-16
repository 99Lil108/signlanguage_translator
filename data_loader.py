import os.path

import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import config

DEVICE = config.device


def subsequent_mask(size):
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class decode_book:
    def __init__(self, vocab_path, oom=-10000, unk='<unk>', pad_idx=0, bos_idx=2, eos_idx=3):
        assert os.path.exists(vocab_path)
        self.id_to_word = {}
        self.word_to_id = {}
        self.oom = oom
        self.unk = unk
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        with open(vocab_path, 'r', encoding='utf8') as f:
            items = f.readline()
            while items:
                word, id = items.strip().split(',')
                self.id_to_word[int(id)] = word
                self.word_to_id[word] = int(id)

                items = f.readline()

    def decode_word_to_id(self, word):
        if self.word_to_id[word]:
            return self.word_to_id[word]
        else:
            return self.oom

    def decode_id_to_word(self, id):
        if self.id_to_word[id]:
            return self.id_to_word[id]
        else:
            return self.unk

    def decode_sentence_to_ids(self, sentence):
        return [self.word_to_id[word] if self.word_to_id[word] else self.oom for word in sentence]

    def decode_ids_to_sentence(self, ids):
        return [self.id_to_word[id] if self.id_to_word[id] else self.unk for id in ids]


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class mappingDataset(Dataset):
    def __init__(self, data_path, vocab_path, meta_data_path):
        self.code_book = decode_book(vocab_path)
        self.id_mapper_seqLen = {}
        self.sentence_list = []
        self.ids_list = []
        self.load_data = self.load_data(data_path, meta_data_path)
        self.PAD = self.code_book.pad_idx  # 0
        self.BOS = self.code_book.bos_idx  # 2
        self.EOS = self.code_book.eos_idx  # 3

    def load_data(self, data_path, meta_data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            words = f.readline()
            while words:
                ids = f.readline().strip().split(',')
                words = words.strip().split(',')
                self.ids_list.append(ids)
                self.sentence_list.append(words)
                words = f.readline()

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(None, tgt_text, batch_input, batch_target, self.PAD)
