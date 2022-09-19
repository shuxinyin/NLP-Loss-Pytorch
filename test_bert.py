import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer

from bert_model.dataloader import TextDataset, BatchTextCall
from bert_model.model import MultiClass

from adversarial_training.FGSM import FGSM
from adversarial_training.FGM import FGM
from adversarial_training.PGD import PGD
from adversarial_training.FreeAT import FreeAT


def choose_bert_type(path, bert_type="tiny_albert"):
    """
    choose bert type for chinese, tiny_albert or macbert（bert）
    return: tokenizer, model
    """

    if bert_type == "albert":
        model_config = BertConfig.from_pretrained(path)
        model = AlbertModel.from_pretrained(path, config=model_config)
    elif bert_type == "bert" or bert_type == "roberta":
        model_config = BertConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, config=model_config)
    else:
        model_config, model = None, None
        print("ERROR, not choose model!")

    return model_config, model


def choose_attack_type(model, attack_type="FGM"):
    if attack_type == 'FGSM':
        attack_model = FGSM(model)
    elif attack_type == 'FGM':
        attack_model = FGM(model)
    elif attack_type == 'PGD':
        attack_model = PGD(model)
    elif attack_type == 'FreeAT':
        attack_model = FreeAT(model)
    return attack_model


def evaluation(model, test_dataloader, loss_func, label2ind_dict, save_path, valid_or_test="test"):
    # model.load_state_dict(torch.load(save_path))

    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for ind, (token, segment, mask, label) in enumerate(test_dataloader):
        token = token.cuda()
        segment = segment.cuda()
        mask = mask.cuda()
        label = label.cuda()

        with torch.no_grad():
            out = model(token, segment, mask)
        loss = loss_func(out, label)
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        predic = torch.max(out.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, label)
        predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, total_loss / len(test_dataloader)


def train(config):
    label2ind_dict = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5, 'politics': 6,
                      'sports': 7, 'game': 8, 'entertainment': 9}

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.backends.cudnn.benchmark = True
    loss_func = F.cross_entropy

    # load_data(os.path.join(data_dir, "cnews.train.txt"), label_dict)

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)
    train_dataset_call = BatchTextCall(tokenizer, max_len=config.sent_max_len)

    train_dataset = TextDataset(os.path.join(config.data_dir, "train.txt"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=10,
                                  collate_fn=train_dataset_call)

    test_dataset = TextDataset(os.path.join(config.data_dir, "test.txt"))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size // 2, shuffle=False, num_workers=10,
                                 collate_fn=train_dataset_call)

    model_config, bert_encode_model = choose_bert_type(config.pretrained_path, bert_type=config.bert_type)
    multi_classification_model = MultiClass(bert_encode_model, model_config,
                                            num_classes=10, pooling_type=config.pooling_type)
    multi_classification_model.cuda()
    # multi_classification_model.load_state_dict(torch.load(config.save_path))
    AT_Model = choose_attack_type(multi_classification_model, attack_type=config.AT_type)

    num_train_optimization_steps = len(train_dataloader) * config.epoch
    param_optimizer = list(multi_classification_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=config.lr)
    # optimizer = transformers.AdamW(multi_classification_model.parameters(), lr=config.lr)

    # loss_func = F.cross_entropy
    loss_total, top_acc = [], 0
    losses, acc_list, eval_loss_list, time_list = [], [], [], []

    for epoch in range(config.epoch):
        multi_classification_model.train()
        start_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="Training epoch{epoch}".format(epoch=epoch))
        for i, (token, segment, mask, label) in enumerate(tqdm_bar):
            token = token.cuda()
            segment = segment.cuda()
            mask = mask.cuda()
            label = label.cuda()
            outputs, loss = AT_Model.train_bert(token, segment, mask, label, optimizer, attack=config.use_attack)

            loss_total.append(loss.detach().item())
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (
            epoch, np.mean(loss_total), time.time() - start_time))
        losses.append(np.mean(loss_total))
        time_list.append(time.time() - start_time)
        time.sleep(0.5)

        acc, loss_test = evaluation(multi_classification_model,
                                    test_dataloader, loss_func, label2ind_dict,
                                    config.save_path)
        print("Accuracy: %.4f Loss in test %.4f" % (acc, loss_test))
        acc_list.append(acc)
        eval_loss_list.append(loss_test)

    # plt.plot(losses)
    # plt.plot(acc_list)
    # plt.plot(eval_loss_list)
    # plt.show()

    result_file = f"out/ATType{config.AT_type}_UseAT{bool(config.use_attack)}.csv"
    df = pd.DataFrame(
        {'train_loss': losses, 'time': time_list, 'acc': acc_list, 'eval_loss': eval_loss_list})
    df.to_csv(result_file, index=False, sep='\t', encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert classification')
    parser.add_argument("--data_dir", type=str, default="./data/THUCNews/news")
    parser.add_argument("--save_path", type=str, default="../ckpt/bert_classification")
    parser.add_argument("--pretrained_path", type=str, default="/data/Learn_Project/Backup_Data/bert_chinese",
                        help="pre-train model path")
    parser.add_argument("--bert_type", type=str, default="bert", help="bert or albert")
    parser.add_argument("--AT_type", type=str, default="FGM", help="FGM, PGD or FreeAT")
    parser.add_argument("--use_attack", type=int, default=1, help="1 represents use")
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--pooling_type", type=str, default="first-last-avg")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sent_max_len", type=int, default=44)
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Set this flag true if you are using an uncased model.")
    args = parser.parse_args()

    print(args)
    train(args)
