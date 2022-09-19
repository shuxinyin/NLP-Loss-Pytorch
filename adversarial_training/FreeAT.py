import torch
import torch.nn.functional as F



class FreeAT:
    def __init__(self, model, epsilon=0.8, k=3, emb_name='embedding.'):
        self.model = model
        self.emb_backup = {}
        self.epsilon = epsilon
        self.K = k  # attack times
        self.emb_name = emb_name  # embedding layer name want to attack
        self.backup_emb()

    def train(self, input_data, labels, optimizer):
        ''' define process of training here according to your model define
        '''
        pass

    def train_bert(self, token, segment, mask, labels, optimizer, attack=True):
        ''' add disturbance in training
        '''
        outputs = self.model(token, segment, mask)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        if attack:
            for t in range(self.K):
                outputs = self.model(token, segment, mask)
                self.model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.attack_emb(backup=False)  # accumulate projected disturb in embedding

        return outputs, loss

    def attack_param(self, name, param):
        '''add  disturbance
           FreeAT Format:
            r[t+1] = r[t] + epsilon * sign(grad)
            r_at = epsilon * np.sign(param.grad)
        '''
        norm = torch.norm(param.grad)
        if norm != 0:
            r_at = self.epsilon * param.grad / norm
            param.data.add_(r_at)
            param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        ''' projected  disturbance like disturb cropping inside the pale (-eps, eps)
        '''
        r = param_data - self.emb_backup[param_name]  # compute  disturbance
        if torch.norm(r) > self.epsilon:  # disturbance cropping inside the pale (-eps, eps)
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def attack_emb(self, backup=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if backup:  # backup embedding
                    self.emb_backup[name] = param.data.clone()
                self.attack_param(name, param)

    def backup_emb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()

    def restore_emb(self):
        '''recover embedding'''
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
