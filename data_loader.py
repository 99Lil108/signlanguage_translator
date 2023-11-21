import os.path
import random

import torch
import cv2
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

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
                word, id = items.strip().split(' ')
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

    def __init__(self, trg_text, src, src_mask, trg=None, pad=0):
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        # self.src_mask = (src != pad).unsqueeze(-2)
        self.src_mask = src_mask
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
    def __init__(self, data_path, vocab_path, meta_data_path, id_to_seqLen_mapper, config=None):
        self.code_book = decode_book(vocab_path)

        self.meta_data_path = meta_data_path

        self.data_list = []  # item :[ids sentence total_seq]
        self.mapper = id_to_seqLen_mapper

        self.load_data = self.load_data(data_path, meta_data_path)
        self.PAD = self.code_book.pad_idx  # 0
        self.BOS = self.code_book.bos_idx  # 2
        self.EOS = self.code_book.eos_idx  # 3

        if config != None:
            self.seq_max_len = config.seq_max_len
            self.image_size = config.image_size
            self.image_padding = config.image_padding
            self.image_bos = config.image_bos
            self.image_eos = config.image_eos
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0 / 255.0,))
            ])

    def load_data(self, data_path, meta_data_path):
        for id in self.code_book.id_to_word.keys():
            if id not in self.mapper:
                video_capture = cv2.VideoCapture(os.path.join(os.getcwd(), meta_data_path, f'{str(id)}.mp4'))
                self.mapper[id] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                video_capture.release()

        with open(data_path, 'r', encoding='utf-8') as f:
            words = f.readline()
            while words:
                ids = [int(id_s) for id_s in f.readline().strip().split(',')]
                words = words.strip().split(',')
                item = [ids, words]
                seqLen = sum([self.mapper[id] for id in ids])
                item.append(seqLen)
                self.data_list.append(item)
                words = f.readline()

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        ids_bs = []
        sentence_bs = []
        seqLen_bs = []
        max_len = 0
        for sample in batch:
            ids_bs.append(sample[0])
            sentence_bs.append(sample[1])
            seqLen_bs.append(sample[2])
            if sample[2] > max_len:
                max_len = sample[2]

        tgt_text = [''.join(sent) for sent in sentence_bs]

        tgt_tokens = [[self.BOS] + ids + [self.EOS] for ids in ids_bs]
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        max_len = self.seq_max_len if max_len > self.seq_max_len else max_len
        batch_input = []
        batch_src_mask = []
        for idx, seq_len in enumerate(seqLen_bs):
            tensor_list, step, cur, discarded = [torch.full((1, self.image_size, self.image_size), self.image_bos,
                                                            dtype=torch.float32)], 1, 0, 0
            if seq_len > self.seq_max_len - 2:
                step = seq_len // (self.seq_max_len - 2)
                tensor_len = self.seq_max_len
            else:
                tensor_len = seq_len

            random_idx = [random.randint(0, step - 1) for _ in range(self.seq_max_len - 2)]

            cap = None
            while len(tensor_list) < tensor_len - 1:
                if cap is None:
                    try:
                        cap = cv2.VideoCapture(
                            os.path.join(self.meta_data_path, f'{str(ids_bs[idx][cur])}.mp4'))
                    except:
                        cur += 1
                        cap = None
                        continue
                tensor_idx = len(tensor_list) - 1
                frame_index = tensor_idx * step + random_idx[tensor_idx] - discarded

                if frame_index >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # if frame_index >= self.mapper[ids_bs[idx][cur]]:
                    cur += 1
                    discarded += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # discarded += self.mapper[ids_bs[idx][cur]]
                    cap.release()
                    cap = None
                else:
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = cap.read()
                        if ret:
                            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            tensor_list.append(self.transform(gray_image))
                        else:
                            raise RuntimeError
                    except:
                        # print(frame_index)
                        # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        pass

            if cap is not None:
                cap.release()

            if len(tensor_list) < max_len - 1:
                tensor_list.extend(
                    [torch.full((1, self.image_size, self.image_size), self.image_padding, dtype=torch.float32) for _ in
                     range(max_len - 1 - len(tensor_list))])

            tensor_list.append(
                torch.full((1, self.image_size, self.image_size), self.image_eos, dtype=torch.float32))

            mask = torch.zeros(max_len, dtype=torch.bool,device=DEVICE)
            mask[:len(tensor_list)] = True

            tensor_list = torch.stack(tensor_list)
            batch_input.append(tensor_list)
            batch_src_mask.append(mask)

        batch_input = torch.stack(batch_input)
        batch_src_mask = torch.stack(batch_src_mask).to(DEVICE).unsqueeze(-2)

        return Batch(tgt_text, batch_input, batch_src_mask, batch_target, self.PAD)
