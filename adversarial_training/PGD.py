import torch
import torch.nn.functional as F


class PGD:
    def __init__(self, model, epsilon=1.0, alpha=0.3, k=3, emb_name='embedding.'):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.K = k  # PGD attack times
        self.emb_name = emb_name

    def train(self, input_data, labels, optimizer):
        ''' define process of training here according to your model define
        '''
        pass

    def train_bert(self, token, segment, mask, labels, optimizer, attack=True):
        ''' a advertisement training demo for bert
        '''
        outputs = self.model(token, segment, mask)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        if attack:
            self.backup_grad()
            for t in range(self.K):
                self.attack_embedding(backup=(t == 0))
                if t != self.K - 1:
                    self.model.zero_grad()
                else:
                    self.restore_grad()
                outputs = self.model(token, segment, mask)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
            self.restore_embedding()  # recover embedding
        optimizer.step()
        self.model.zero_grad()

        return outputs, loss

    def attack_param(self, name, param):
        '''add  disturbance
            PGD: r = epsilon * grad / norm(grad)
        '''
        norm = torch.norm(param.grad)
        if norm != 0 and not torch.isnan(norm):
            r_at = self.alpha * param.grad / norm
            param.data.add_(r_at)
            param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        ''' projected  disturbance like parameter cropping inside the pale
        '''
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def attack_embedding(self, backup=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if backup:  # backup embedding
                    self.emb_backup[name] = param.data.clone()
                self.attack_param(name, param)

    def backup_embedding(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()

    def restore_embedding(self):
        '''recover embedding'''
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        '''recover grad back upped
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]
