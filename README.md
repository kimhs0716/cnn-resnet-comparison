# CNN vs ResNet Comparison on CIFAR-10

Residual connection이 정확도뿐 아니라 학습 안정성 / 수렴 속도 / 깊이 확장성에서 어떤 구조적 차이를 만드는지 실험적으로 분석하는 프로젝트

## Research Question

> Residual connection은 단순히 정확도를 올리는가,
> 아니면 학습 안정성과 깊이 확장성에서 구조적으로 다른가?

## Experiment Plan

| ID | Model | Augmentation | Purpose |
|----|-------|-------------|---------|
| E1 | PlainCNN | - | baseline 하한 |
| E2 | ResNet | - | residual 효과 순수 분리 |
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
| E5 | PlainCNN-deep | - | Flip+Crop | - | - | - |
| E6 | ResNet-deep | - | Flip+Crop | - | - | - |

## Project Structure

```
cnn-resnet-comparison/
├── configs.yaml          # 학습 설정 (device, batch size, epochs, seed 등)
├── requirements.txt
├── data/                 # CIFAR-10 자동 다운로드
├── scripts/
│   ├── main.py           # 학습 실행 파일
│   └── main.ipynb        # 학습 실행 노트북 (실험용)
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

> Intel GPU(XPU) 환경 기준. `configs.yaml`의 `device` 값을 `cpu` / `cuda`로 바꾸면 각각 CPU / CUDA 환경에서도 동작.

## Run

```bash
python scripts/main.py
```

CIFAR-10 데이터는 첫 실행 시 `data/` 디렉토리에 자동으로 다운로드

## Environment

- Python 3.x
- PyTorch 2.9.1+xpu (Intel GPU / XPU)
- torchvision 0.24.1+xpu

## Reference

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016
