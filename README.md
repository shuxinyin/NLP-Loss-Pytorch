Implementation of some unbalanced loss for NLP task like focal_loss, dice_loss, DSC Loss, GHM Loss et.al and adversarial
training like FGM, FGSM, PGD, FreeAT.

### Loss Summary

Here is a loss implementation repository included unbalanced loss

|    Loss Name     |                                                                paper                                                                 | Notes |
|:----------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:-----:|
| Weighted CE Loss | [UNet Architectures in Multiplanar Volumetric Segmentation -- Validated on Three Knee MRI Cohorts](https://arxiv.org/abs/2203.08194) |       |
|    Focal Loss    |                              [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)                               |       |
|    Dice Loss     |       [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)       |       |
|     DSC Loss     |                           [Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/pdf/1911.02855.pdf)                            |       |
|     GHM Loss     |           [Gradient Harmonized Single-stage Detector](https://www.aaai.org/ojs/index.php/AAAI/article/download/4877/4750)            |       |
| Label Smoothing  |                               [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629.pdf)                                |       |

#### How to use?

You can find all the loss usage information in test_loss.py.

Here is a simple demo of usage:

```python
import torch
from unbalanced_loss.focal_loss import MultiFocalLoss

batch_size, num_class = 64, 10
Loss_Func = MultiFocalLoss(num_class=num_class, gamma=2.0, reduction='mean')

logits = torch.rand(batch_size, num_class, requires_grad=True)  # (batch_size, num_classes)
targets = torch.randint(0, num_class, size=(batch_size,))  # (batch_size, )

loss = Loss_Func(logits, targets)
loss.backward()
```

### Adversarial Training Summary

Here is a Summary of Adversarial Training implementation.   
you can find more details in adversarial_training/README.md

| Adversarial Training |                                                 paper                                                 | Notes |
|:--------------------:|:-----------------------------------------------------------------------------------------------------:|:-----:|
|         FGM          |                     [Fast Gradient Method](https://arxiv.org/pdf/1605.07725.pdf)                      |       |
|         FGSM         |                     [Fast Gradient Sign Method](https://arxiv.org/abs/1412.6572)                      |       |
|         PGD          | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf) |       |
|        FreeAT        |                   [Free Adversarial Training](https://arxiv.org/pdf/1904.12843.pdf)                   |       |
|        FreeLB        |            [Free Large Batch Adversarial Training](https://arxiv.org/pdf/1909.11764v5.pdf)            |       |

#### How to use?

**You can find a simple demo for bert classification in test_bert.py.**

Here is a simple demo of usage:  
You just need to rewrite train function according to input for your model in file PGD.py, then you can use adversarial
training like below.

```python
import transformers
from model import bert_classification
from adversarial_training.PGD import PGD

batch_size, num_class = 64, 10
# model = your_model()
model = bert_classification()
AT_Model = PGD(model)
optimizer = transformers.AdamW(model.parameters(), lr=0.001)

# rewrite your train function in pgd.py
outputs, loss = AT_Model.train_bert(token, segment, mask, label, optimizer)
```

#### Adversarial Training Results

here are some results tested on THNews classification task based on bert.   
you can find run the code as below:
> cd scripts  
> sh run_at.sh

|  Adversarial Training  | Time Cost(s/epoch ) | best_acc |
|:----------------------:|:-------------------:|:--------:|
| Normal(not add attack) |        23.77        |  0.773   |
|          FGSM          |        45.95        |  0.7936  |
|          FGM           |        47.28        |  0.8008  |
|        PGD(k=3)        |        87.50        |  0.7963  |
|      FreeAT(k=3)       |        93.26        |  0.7896  |