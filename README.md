# Adversarially Robust Generalization Just Requires More Unlabeled Data

This is the repository for the paper Adversarially Robust Generalization Just Requires More Unlabeled Data submitted to NeurIPS 2019 ([paper link](https://arxiv.org/abs/1906.00555)).

## Code Files

| File | Information |
| ------ | ------ |
| adv_train.py | Original adversarial training |
| transductive.py | Transductive setting |
| 5k10k.py | 5k/10k experiment |
| model.py | WResNet-32 model |
| utils.py | Helper functions for train/test|

## Checkpoint

Checkpoints can be downloaded [here](https://1drv.ms/u/s!AoKLjEkI6Z_qxTTjva_5l2aGPQ_f?e=rWRheu). Use the following code to load the checkpoints:
```bash
checkpoint = torch.load('checkpoint.t7')
net.load_state_dict(checkpoint['net'])
```

The following checkpoints are included.  

| File | Information |
| ------ | ------ |
| 5k-{0.0,0.1,0.2,0.3}.t7 | 5k experiment with lambda = 0.0,0.1,0.2,0.3|
| 10k-{0.0,0.1,0.2,0.3}.t7 | 10k experiment with lambda = 0.0,0.1,0.2,0.3 
| transductive.t7 | Transductive setting with lambda = 0.125 |
| pgd7_adv_train.t7 | Original adversarial training |
