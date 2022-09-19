import torch
import torch.nn.functional as F


class FGM:
    def __init__(self, model, emb_name='embedding.'):
        self.model = model
        self.emb_backup = {}  # restore embedding parameters
        self.epsilon = 1.0
        self.emb_name = emb_name

    def train(self, input_data, labels, optimizer):
        ''' define process of training here according to your model define
        '''
        pass

    def train_bert(self, token, segment, mask, labels, optimizer, attack=False):
        ''' a advertisement training demo for bert
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

    def attack_embedding(self, backup=True):
        ''' add add disturbance in embedding layer you want
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if backup:  # store parameter
                    self.emb_backup[name] = param.data.clone()

                self._add_disturbance(name, param)  # add disturbance

    def restore_embedding(self):
        '''recover embedding backup before
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def _add_disturbance(self, name, param):
        ''' add disturbance
        '''
        norm = torch.norm(param.grad)
        if norm != 0:
            r_at = self.epsilon * param.grad / norm
            param.data.add_(r_at)
