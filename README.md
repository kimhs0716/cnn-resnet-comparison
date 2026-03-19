# CNN vs ResNet Comparison on CIFAR-10

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

| ID | Model | Params | Aug | Epochs | Best Val Acc | Test Acc |
|----|-------|--------|-----|--------|-------------|----------|
| E1 | PlainCNN | 289K | - | 30 | 83.44% (ep.28) | **82.60%** |
| E2 | ResNet | 300K | - | 30 | 78.16% (ep.21) | **77.89%** |
| E3 | PlainCNN | 289K | Flip+Crop | 30 | 86.42% (ep.30) | **86.25%** |
| E4 | ResNet | 300K | Flip+Crop | 30 | 84.16% (ep.23) | **83.30%** |
| E5 | PlainCNN-deep | 678K | Flip+Crop | 30 | 88.52% (ep.30) | **87.48%** |
| E6 | ResNet-deep | 687K | Flip+Crop | 30 | 86.58% (ep.23) | **85.36%** |

## Project Structure

```
cnn-resnet-comparison/
├── configs.yaml          # 학습 설정 (device, batch size, epochs, seed 등)
├── requirements.txt
├── data/                 # CIFAR-10 자동 다운로드
├── scripts/
│   └── main.py           # 학습 실행 파일
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

- Python 3.x
- PyTorch 2.9.1
- torchvision 0.24.1

## Reference

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016
