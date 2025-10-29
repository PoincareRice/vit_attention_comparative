# 🧠 ViT Attention Comparative  
### Comparative Analysis of Advanced Attention Mechanisms in Vision Transformers  
**비전 트랜스포머(ViT)에서 고급 어텐션 메커니즘(Entmax, Sparsemax 등)을 비교 분석하는 프로젝트**

-

## 📘 Overview | 개요

This repository provides a complete implementation for comparing **advanced attention mechanisms** (such as *Entmax* and *Sparsemax*) in **Vision Transformers (ViT)**.  
It supports training, evaluation, and visualization of attention maps on **CIFAR-10**.

이 저장소는 **Vision Transformer(ViT)** 에서 **Entmax, Sparsemax 등 고급 어텐션 메커니즘**을 비교 실험하기 위한 전체 구현을 제공합니다.  
**CIFAR-10** 데이터셋을 기반으로 학습, 평가, 어텐션 맵 시각화를 지원합니다.

---

## 📂 Project Structure | 프로젝트 구조

```
VIT_ATTENTION_COMPARATIVE/
├── bash_folder/               # Shell scripts for training & evaluation | 학습/평가 실행 스크립트
│   ├── train.bash
│   └── evaluate.bash
├── configs/                   # Experiment and model configuration files | 설정 파일
│   ├── config.yaml
│   └── experiment1.yaml
├── data/                      # Dataset storage (CIFAR-10) | 데이터 저장 경로
│   ├── external/              # Raw/external data | 원본 데이터
│   ├── processed/
│   └── raw/
├── experiments/               # Experiment results | 실험 결과 저장
│   └── exp_001/
│       ├── checkpoints/       # Model checkpoints | 모델 가중치 저장
│       ├── logs/
│       └── results/
├── src/
│   ├── data/                  # Dataset loaders | 데이터셋 로더
│   ├── evaluation/            # Evaluation scripts | 평가 코드
│   ├── models/                # Model definitions | 모델 정의
│   │   ├── attention/
│   │   │   ├── base_attention.py
│   │   │   ├── entmax_attention.py
│   │   │   └── sparsemax_attention.py
│   │   └── vit_custom.py
│   ├── training/              # Training scripts | 학습 코드
│   └── utils/                 # Utility functions | 유틸리티 함수
├── tests/                     # Unit tests | 테스트 코드
└── requirements.txt
```
---

## ⚙️ Installation | 설치 방법

```bash
# Clone repository | 저장소 복제
git clone https://github.com/PoincareRice/vit_attention_comparative.git
cd VIT_ATTENTION_COMPARATIVE

# Create virtual environment (optional) | 가상환경 생성 (선택)
conda create -n vit-env python=3.10 -y
conda activate vit-env
#ubuntu22.04
python3.10 -m venv vit-env
source ./vit-env/bin/activate

# Install dependencies | 필요한 패키지 설치
pip install -r requirements.txt
```
---
## 🚀 Usage | 사용 방법

### Training | 학습 실행
```bash
bash ./bash_folder/train.bash
```
Trains a ViT model using the selected attention mechanism (softmax, entmax, or sparsemax).
Saves checkpoints under experiments/<exp_name>/checkpoints/.

선택한 어텐션 메커니즘(softmax, entmax, sparsemax)으로 ViT 모델을 학습합니다.
결과 가중치는 experiments/<exp_name>/checkpoints/ 폴더에 저장됩니다.

### Evaluation | 평가 실행
```bash
bash ./bash_folder/evaluate.bash
```
Evaluates a saved checkpoint and visualizes attention maps.
Results are stored under experiments/<exp_name>/results/.

저장된 체크포인트를 불러와 모델을 평가하고 어텐션 맵을 시각화합니다.
결과는 experiments/<exp_name>/results/ 폴더에 저장됩니다.

