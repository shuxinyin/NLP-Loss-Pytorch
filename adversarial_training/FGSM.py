import numpy as np
import torch.nn.functional as F


class FGSM:
    def __init__(self, model, epsilon=0.05, emb_name='embedding.'):
        self.model = model
        self.emb_backup = {}
        self.epsilon = epsilon
        self.emb_name = emb_name

    def train(self, input_data, labels, optimizer):
        ''' define process of training here according to your model define
        '''
        pass

    def train_bert(self, token, segment, mask, labels, optimizer, attack=False):
        ''' add disturbance in training
        '''
        outputs = self.model(token, segment, mask)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        if attack:
            self.attack_embedding()
            outputs = self.model(token, segment, mask)
            loss = F.cross_entropy(outputs, labels)
            # self.model.zero_grad()  # compute advertise samples' grad only
            loss.backward()
            self.restore_embedding()  # recover
        optimizer.step()
        self.model.zero_grad()

        return outputs, loss

    def attack_param(self, name, param):
        # r_at = epsilon * sign(grad)
        r_at = self.epsilon * np.sign(param.grad)
        param.data.add_(r_at)

    def attack_embedding(self, backup=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if backup:
                    self.emb_backup[name] = param.data.clone()
                # attack embedding
                self.attack_param(name, param)

    def restore_embedding(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
