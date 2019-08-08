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

## Cite
Please cite our paper with the following BibTeX entry:
```
@article{DBLP:journals/corr/abs-1906-00555,
  author    = {Runtian Zhai and
               Tianle Cai and
               Di He and
               Chen Dan and
               Kun He and
               John E. Hopcroft and
               Liwei Wang},
  title     = {Adversarially Robust Generalization Just Requires More Unlabeled Data},
  journal   = {CoRR},
  volume    = {abs/1906.00555},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.00555},
  archivePrefix = {arXiv},
  eprint    = {1906.00555},
  timestamp = {Thu, 13 Jun 2019 13:36:00 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906-00555},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
