# InfoBatch

Not official implementation of InfoBatch. This repository is for study and research purpose.

## Introduction

[Paper](https://arxiv.org/abs/2303.04947)
[Paper Review with Korean](Paper_review.md)

## To do

[x] Cifar10, Cifar100
[] ImageNet
[] Diffusion Model

## Experiment
### Cifar 10

Train ResNet50 with Infobatch, 

```
python examples/cifar10_train.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 -- ratio 0.5  --use_info_batch
```

Train ResNet50 without Infobatch, 

```
python examples/cifar10_train.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 -- ratio 0.5
```

### Cifar 100

Train ResNet50 with Infobatch, 

```
python examples/cifar100_train.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 -- ratio 0.5  --use_info_batch
```

Train ResNet50 without Infobatch, 

```
python examples/cifar100_train.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 -- ratio 0.5
```

## Results

### Cifar 10 

| Model | Infobatch | Accuracy | Time(H) |
|-------|-----------|----------|------|
| ResNet18 | X | 95.35% | 2.93H
| ResNet18 | O | 95.49% | 1.88H
| ResNet50 | X | 95.53% | 6.65H
| ResNet50 | O | 95.57% | 4.25H

### Cifar 100

| Model | Infobatch | Accuracy | Time(H) |
|-------|-----------|----------|------|
| ResNet18 | X | 79.92% | 2.93 H
| ResNet18 | O | 79.55% | 2.03 H
| ResNet50 | X | 81.39% | 6.67 H
| ResNet50 | O | 80.55% | 4.60 H

## Reference

[InfoBatch Official code](https://github.com/NUS-HPC-AI-Lab/InfoBatch/tree/master)
[Junia3's github](https://github.com/junia3/InfoBatch/tree/main)