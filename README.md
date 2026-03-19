# CNN vs ResNet Comparison on CIFAR-10/100

Residual Block을 사용하는 ResNet 모델과 일반 CNN 모델간의 성능상 차이 비교

## Research Question

> Residual connection은 단순히 정확도를 올리는가,
> 아니면 학습 안정성과 깊이 확장성에서 구조적으로 다른가?

## Experiment Plan

| ID | Model | Augmentation | Purpose |
|----|-------|-------------|---------|
| E1 | PlainCNN | - | baseline 기준 |
| E2 | ResNet | - | residual 효과 비교 |
| E3 | PlainCNN | Flip + Crop | augmentation 효과 비교 |
| E4 | ResNet | Flip + Crop | augmentation 효과 비교 |
| E5 | PlainCNN-deep | Flip + Crop | degradation 확인 |
| E6 | ResNet-deep | Flip + Crop | 깊이 확장성 확인 |

## Results

### CIFAR-10

| ID | Model | Params | Aug | Epochs | Best Epoch | Test Acc |
|----|-------|--------|-----|--------|------------|----------|
| E1 | PlainCNN | 298K | - | 50 | 14 | **80.10%** |
| E2 | ResNet | 309K | - | 50 | 9 | **78.01%** |
| E3 | PlainCNN | 298K | Flip+Crop | 50 | 29 | **85.28%** |
| E4 | ResNet | 309K | Flip+Crop | 50 | 44 | **86.44%** |
| E5 | PlainCNN-deep | 1074K | Flip+Crop | 50 | 44 | **86.83%** |
| E6 | ResNet-deep | 1085K | Flip+Crop | 50 | 34 | **89.73%** |

### CIFAR-100

| ID | Model | Params | Aug | Epochs | Best Epoch | Test Acc |
|----|-------|--------|-----|--------|------------|----------|
| E1 | PlainCNN | 310K | - | 50 | 25 | **54.33%** |
| E2 | ResNet | 320K | - | 50 | 13 | **49.63%** |
| E3 | PlainCNN | 310K | Flip+Crop | 50 | 41 | **57.70%** |
| E4 | ResNet | 320K | Flip+Crop | 50 | 47 | **59.80%** |
| E5 | PlainCNN-deep | 1086K | Flip+Crop | 50 | 27 | **60.77%** |
| E6 | ResNet-deep | 1096K | Flip+Crop | 50 | 23 | **62.63%** |

## Conclusion

**Augmentation 없이는 PlainCNN이 우세하나, augmentation 적용 시 ResNet이 역전됨. 깊이 확장 효과는 데이터셋에 따라 다르게 나타남.**

| 조건 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| No aug | PlainCNN 우세 | PlainCNN 우세 |
| Aug (shallow) | ResNet 우세 (+1.16%p) | ResNet 우세 (+2.10%p) |
| Aug (deep) | ResNet 우세 (+2.90%p) | ResNet 우세 (+1.86%p) |

- **Augmentation 없이는 overfitting이 지배적**이라 PlainCNN이 더 빠르게 수렴
- **Augmentation 적용 시 ResNet이 역전** — 두 데이터셋 모두에서 일관된 패턴
- **CIFAR-10에서는 깊이가 증가할수록 ResNet의 우위가 커짐** (+1.16%p → +2.90%p) — residual connection의 degradation 방지 효과
- **CIFAR-100에서는 shallow에서 ResNet 우위가 더 큼** (+2.10%p → +1.86%p) — 깊이 확장에 따른 격차 증가는 CIFAR-10에서만 관찰됨
- **방향성(ResNet 우위)은 두 데이터셋 모두에서 일관됨**

### 한계

- 로컬 환경(Intel XPU)의 oneDNN 비결정성으로 인해 실행마다 수치 차이가 존재함
- 로컬(XPU)과 Colab(CUDA) 간 결과도 다를 수 있음 — 개별 수치보다 방향성을 중심으로 해석해야 함

## Project Structure

```
cnn-resnet-comparison/
├── configs.yaml          # 학습 설정 (device, batch size, epochs, seed 등)
├── requirements.txt
├── data/                 # CIFAR-10/100 자동 다운로드
├── scripts/
│   └── main.py           # 학습 실행 파일
├── results/              # 학습 곡선 및 metrics
│   ├── CIFAR10/  
│   └── CIFAR100/  
└── src/
    ├── data.py           # DataLoader (train / val / test)
    ├── models.py         # PlainCNN, ResNet, ResidualBlock
    ├── trainer.py        # train / evaluate 함수
    └── utils.py          # device 설정, seed 고정, 시각화
```

## Setup

Python 3.12 버전 사용을 권장합니다.

```bash
pip install -r requirements.txt
```


Intel GPU 환경을 지원합니다.
```bash
pip install -r requirements_xpu.txt
```
`configs.yaml`의 `device`를 `xpu`로 수정


## Run

```bash
python scripts/main.py
```

CIFAR-10 / CIFAR-100 데이터는 첫 실행 시 `data/` 디렉토리에 자동으로 다운로드

## Environment

- Python 3.12
- PyTorch 2.9.1
- torchvision 0.24.1

## Reference

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016
