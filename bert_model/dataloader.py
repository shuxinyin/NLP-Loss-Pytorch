import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AlbertModel, BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, AutoModelForMaskedLM


def load_data(path):
    train = pd.read_csv(path, header=0, sep='\t', names=["text", "label"])
    print(train.shape)

    texts = train.text.to_list()
    labels = train.label.map(int).to_list()
    # label_dic = dict(zip(train.label.unique(), range(len(train.label.unique()))))
    return texts, labels


class TextDataset(Dataset):
    def __init__(self, filepath):
        super(TextDataset, self).__init__()
        self.train, self.label = load_data(filepath)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        text = self.train[item]
        label = self.label[item]
        return text, label


class BatchTextCall(object):
    """call function for tokenizing and getting batch text
    """

    def __init__(self, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, batch_text):
        return self.tokenizer(batch_text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]

        source = self.text2id(batch_text)
        token = source.get('input_ids').squeeze(1)
        mask = source.get('attention_mask').squeeze(1)
        segment = source.get('token_type_ids').squeeze(1)
        label = torch.tensor(batch_label)

        return token, segment, mask, label


if __name__ == "__main__":

    data_dir = "../data/THUCNews/news"
    pretrained_path = "/data/Learn_Project/Backup_Data/bert_chinese"

    label_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8,
                  '财经': 9}

    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    model_config = BertConfig.from_pretrained(pretrained_path)
    model = BertModel.from_pretrained(pretrained_path, config=model_config)

    text_dataset = TextDataset(os.path.join(data_dir, "train.txt"))
    text_dataset_call = BatchTextCall(tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=text_dataset_call)

    for i, (token, segment, mask, label) in enumerate(text_dataloader):
        print(i, token, segment, mask, label)
        out = model(input_ids=token, attention_mask=mask, token_type_ids=segment)
        # loss, logits = model(token, mask, segment)[:2]
        print(out)
        print(out.last_hidden_state.shape)
        break
