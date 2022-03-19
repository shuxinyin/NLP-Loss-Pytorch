import sys

import torch

import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

from unbalanced_loss.focal_loss import MultiFocalLoss, BinaryFocalLoss
from unbalanced_loss.dice_loss_nlp import BinaryDSCLoss, MultiDSCLoss

torch.manual_seed(123)


class CNNModel(nn.Module):

    def __init__(self, num_class, kernel_size=3, padding=1, stride=1):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(*[nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU()])
        self.fc = nn.Linear(32 * 32 * 16, num_class)  # flatten length * width * channels

    def forward(self, data):
        output = self.model(data)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def choose_loss(num_class, loss_type):
    '''
    choose loss type
    '''
    if loss_type == "binary_focal_loss":
        data_shape = (16, 3, 32, 32)
        target_shape = (16, )  # [batch, 1]

        datas = (torch.rand(data_shape)).cuda()
        target = torch.randint(0, 2, size=target_shape).cuda()
        Loss = BinaryFocalLoss()

    if loss_type == "multi_class_focal_loss":
        data_shape = (16, 3, 32, 32)  # [batch, channels, width, length]
        target_shape = (16,)  # [batch, ]

        datas = (torch.rand(data_shape)).cuda()
        target = torch.randint(0, num_class, size=target_shape).cuda()
        Loss = MultiFocalLoss(num_class=num_class, gamma=2.0, reduction='mean')

    if loss_type == "binary_dice_loss":  # 重写
        data_shape = (16, 3, 32, 32)
        target_shape = (16, )  # [batch, 1]

        datas = (torch.rand(data_shape)).cuda()
        target = torch.randint(0, 2, size=target_shape).cuda()
        Loss = BinaryDSCLoss()

    if loss_type == "multi_class_dice_loss":
        data_shape = (16, 3, 32, 32)  # [batch, channels, width, length]
        target_shape = (16,)  # [batch,]

        datas = (torch.rand(data_shape)).cuda()
        target = torch.randint(0, num_class, size=target_shape).cuda()
        Loss = MultiDSCLoss(alpha=1.0, smooth=1.0, reduction="mean")

    return datas, target, Loss


def main():
    num_class = 5
    datas, target, Loss = choose_loss(num_class, loss_type="multi_class_focal_loss")
    target = target.long().cuda()
    # print(target.shape, datas.shape)

    model = CNNModel(num_class)
    model = model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    losses = []
    for i in range(32):
        output = model(datas)
        loss = Loss(output, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
