import sacrebleu
import tqdm
import sentencepiece

import config

import torch

def chinese_tokenizer_load():
    sp_chn = sentencepiece.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    return sp_chn


def evaluate(data, model, mode='dev', use_beam=True):
    sp_chn = chinese_tokenizer_load()
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
                                               config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
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
