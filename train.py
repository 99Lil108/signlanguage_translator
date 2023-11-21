import argparse
import logging
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from model import transformer
from loss import labelSmoothing, noamOpt
from utils import evaluate
from data_loader import mappingDataset
import config


def train(bs, work_num, resume, epoch, loss='labelSmoothing'):
    mp.set_start_method('spawn')

    device = config.device
    start_epoch = 0
    best_bleu_score = 0.0

    # model
    model = transformer(config.tgt_vocab_size)
    model.to(device)

    if resume:
        if os.path.exists(config.last_info):
            with open(config.last_info, "r") as file:
                last = json.load(file)
            start_epoch = last["epoch"]
            best_bleu_score = last["best_bleu_score"]
        if os.path.exists(config.last_model):
            model.load_state_dict(torch.load(config.last_model))

    start_epoch += 1
    # dataset
    mapper = {
        config.padding_idx: 1,
        config.eos_idx: 1,
        config.bos_idx: 1,
        config.unk: 0
    }
    train_dataset = mappingDataset(config.data_path, config.vocab_path, config.data_folder, id_to_seqLen_mapper=mapper,
                                   config=config)
    eval_data = None

    # dataloader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs, num_workers=work_num,
                                  collate_fn=train_dataset.collate_fn)

    # loss function
    if loss == 'labelSmoothing':
        criterion = labelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    criterion.to(device=device)

    # optimizer
    opt = noamOpt(512, 1, 10000,
                  torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # start training
    for idx in range(start_epoch, epoch + 1):
        total_tokens = 0.
        total_loss = 0.
        for batch in tqdm(train_dataloader):
            model.train()
            opt.optimizer.zero_grad()
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = criterion(out.contiguous().view(-1, out.size(-1)),
                             batch.trg_y.contiguous().view(-1)) / batch.ntokens
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_tokens += batch.ntokens
        logging.info("Epoch: {}, training loss: {}".format(idx, total_loss / total_tokens))

        # eval
        if eval_data is not None:
            model.eval()
            bleu_score = evaluate(eval_data, model)
            logging.info('Epoch: {}, Bleu Score: {}'.format(epoch, bleu_score))

            if bleu_score > best_bleu_score:
                torch.save(model.state_dict(), config.best_model)
                best_bleu_score = bleu_score
                logging.info(f"-------- save best in the epoch : {epoch} --------")

        torch.save(model.state_dict(), config.last_model)
        last_info = {
            "epoch": idx,
            "best_bleu_score": best_bleu_score
        }
        with open(config.last_info, 'w') as f:
            json.dump(last_info, f, indent=4, sort_keys=True)

    logging.info('finish~')


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', help="batch-size", default=1, type=int)
    parser.add_argument('--work_num', default=1, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--epoch', default=60, type=int)
    args = parser.parse_args()

    assert torch.cuda.is_available()

    train(args.bs, args.work_num, args.resume, args.epoch)
