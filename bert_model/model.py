import torch
from torch import nn

BertLayerNorm = torch.nn.LayerNorm


class MultiClass(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    """

    def __init__(self, bert_encode_model, model_config, num_classes=10, pooling_type='first-last-avg'):
        super(MultiClass, self).__init__()
        self.bert = bert_encode_model
        self.num_classes = num_classes
        self.fc = nn.Linear(model_config.hidden_size, num_classes)
        self.pooling = pooling_type
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(model_config.hidden_size)

    def forward(self, batch_token, batch_segment, batch_attention_mask):
        out = self.bert(batch_token,
                        attention_mask=batch_attention_mask,
                        token_type_ids=batch_segment,
                        output_hidden_states=True)

        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0, :]  # [batch, 768]
        elif self.pooling == 'pooler':
            out = out.pooler_output  # [batch, 768]
        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        else:
            raise "should define pooling type first!"

        out = self.layer_norm(out)
        out = self.dropout(out)
        out_fc = self.fc(out)
        return out_fc


if __name__ == '__main__':
    path = "/data/Learn_Project/Backup_Data/bert_chinese"
    MultiClassModel = MultiClass
    # MultiClassModel = BertForMultiClassification
    multi_classification_model = MultiClassModel.from_pretrained(path, num_classes=10)
    if hasattr(multi_classification_model, 'bert'):
        print("-------------------------------------------------")
    else:
        print("**********************************************")
