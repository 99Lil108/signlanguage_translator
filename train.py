import argparse
import logging
from tqdm import tqdm

import torch.cuda

from model import transformer
from loss import labelSmoothing, noamOpt
import config


def train(bs, work_num, resume, epoch, loss='labelSmoothing'):
    device = torch.device("cuda")
    start_epoch = 0

    start_epoch += 1
    # dataLoader
    train_data = None
    eval_data = None

    # model
    model = transformer(config.tgt_vocab_size)
    model.to(device)

    # loss function
    if loss == 'labelSmoothing':
        criterion = labelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    criterion.to(device=device)

    # optimizer
    opt = noamOpt(512, 1, 10000,
                  torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    best_bleu_score = 0.0
    for idx in range(start_epoch, epoch + 1):
        # train
        total_tokens = 0.
        total_loss = 0.
        for batch in tqdm(train_data):
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
        model.eval()
        total_loss = 0.
        total_tokens = 0.
        with torch.no_grad():
            for batch in tqdm(eval_data):
                out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

                loss = criterion(out.contiguous().view(-1, out.size(-1)),
                                 batch.trg_y.contiguous().view(-1)) / batch.ntokens
                total_loss += loss.item()
                total_tokens += batch.ntokens

        logging.info("Eval loss: {}".format(total_loss / total_tokens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', help="batch-size", default=2)
    parser.add_argument('--work_num', default=2)
    parser.add_argument('--resume', default=False)
    parser.add_argument('--epoch', default=60)
    args = parser.parse_args()

    assert torch.cuda.is_available()

    train(parser.bs, parser.work_num, parser.resum, parser.epoch)
