import sacrebleu
import tqdm

import config
from data_loader import decode_book
from translator import beam_search, batch_greedy_decode

import torch


def get_chn_decode_book(vocab_path, oom=-10000, unk='<unk>'):
    return decode_book(vocab_path, oom=oom, unk=unk)


def evaluate(data, model, mode='dev', use_beam=True):
    device = torch.device('cuda')
    sp_chn = get_chn_decode_book(config.vocab_path)
    trg = []
    res = []
    with torch.no_grad():
        for batch in tqdm(data):
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            decode_result = [h[0] for h in decode_result]

            translation = [sp_chn.decode_ids_to_sentence(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)
