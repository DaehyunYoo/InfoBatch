# LINE VOOM AI Engineer 과제테스트

# Type B: INFOBATCH: LOSSLESS TRAINING SPEED UP BY UNBIASED DYNAMIC DATA PRUNING 실험


## 1. Summary
LINE 과제 테스트를 위한 실험 가이드라인 페이지입니다. 본 실험에서는 1 A100 GPU를 활용하여 진행되었으며, CIFAR-10 데이터셋을 활용하였습니다. Image Classification task에 대해서 ResNet-50을 활용하여 정확도(Accuracy)와 효율성(Training Time)을 비교하였습니다. 

## 2. Experimental Settings
- GPU: A100
- Data: CIFAR-10
- Model: ResNet-50

## 3. 사용방법

Baseline 실행 방법
```
python3 examples/cifar_train.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.0 -seed 0
```

InfoBatch 실행 방법
```
python examples/cifar_train.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 --ratio 0.5 --seed 0
```

## 4. Implementation Details

|         | Configuration |
|:-------------:|:-----------:|
| Epochs  |     200     |
|   Optimizer   |     lars     |
|   Batch size   |     200     |
|   Learning rate   |     0.2     |



# 5. 실험 결과

## 1. CIFAR-10 Dataset(ResNet-50)

|   Method      | Accuracy(%) | Training Time(M) |
|:-------------:|:-----------:|:----------------:|
| Full Dataset  |     95.520     |       231.10        |
|   InfoBatch($r=0.3$)   |     95.500     |       184.66        |
|   InfoBatch($r=0.5$)   |     **95.530**     |       148.44        |
|   InfoBatch($r=0.7$)   |     95.170     |       **113.83**        |


## 2. Comparison of performance and time cost on CIFAR-10.

|   Method      | Full Dataset | InfoBatch |
|:-------------:|:-----------:|:----------------:|
| Accuracy (%)  |     95.520     |       95.210        |
|   Time (h)   |      3.88    |       2.48        |
|   Overhead (h)   |     0.0     |       0.057        |
|   Total (n*h)   |     50000     |       2850        |


## 3. Evaluating Components of InfoBatch
$ r = 0.5 $, $ \delta = 0.875 $

```
1. Rescale을 사용하지 않은 경우:
infobatch.py line 104 주석 처리 
# values.mul_(weights)

2. Annealing을 사용하지 않은 경우:
infobatch.py의 reset 함수 수정

def reset(self):
    np.random.seed(self.iterations)
    self.sample_indices = self.dataset.prune() # if without pruning, 
    self.iter_obj = iter(self.sample_indices)
    self.iterations += 1

```

| $\mathcal{P}$ | Rescale | Annealing | Accuracy (%) |
|:-------------:|:-------:|:---------:|:------------:|
| Random        |         |           |   95.520     |
| Soft          |  |      |    95.600    |
| Soft          | $\checkmark$ |      |     95.340   |
| Soft          |  | $\checkmark$     |     95.590   |
| Soft          | $\checkmark$ |  $\checkmark$    |     95.530   |