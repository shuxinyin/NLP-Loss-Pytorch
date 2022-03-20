

Implementation of some unbalanced loss for NLP task like focal_loss, dice_loss, DSC Loss, GHM Loss et.al
### Summary
Here is a loss implementation repository included unbalanced loss

| Loss Name | paper | Notes |
| :-----:| :----: | :----: |
| Weighted CE Loss | [UNet Architectures in Multiplanar Volumetric Segmentation -- Validated on Three Knee MRI Cohorts](https://arxiv.org/abs/1708.02002) |  |
| Focal Loss | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/2203.08194) |  |
| Dice Loss | [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) |  |
| DSC Loss | [Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/pdf/1911.02855.pdf) |  |
| GHM Loss | [Gradient Harmonized Single-stage Detector](https://www.aaai.org/ojs/index.php/AAAI/article/download/4877/4750) |  |

### How to use?
You can find all the loss usage information in test_loss.py.  
 
Here is a simple demo of usage:
```python
import torch
from unbalanced_loss.focal_loss import MultiFocalLoss

batch_size, num_class = 64, 10
Loss_Func = MultiFocalLoss(num_class=num_class, gamma=2.0, reduction='mean')

logits = torch.rand(batch_size, num_class, requires_grad=True)  # (batch_size, num_classes)
targets = torch.randint(0, num_class, size=(batch_size, ))  # (batch_size, )

loss = Loss_Func(logits, targets)
loss.backward()
```