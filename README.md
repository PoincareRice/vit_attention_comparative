# 🧠 ViT Attention Comparative  
### Comparative Analysis of Advanced Attention Mechanisms in Vision Transformers  
**비전 트랜스포머(ViT)에서 고급 어텐션 메커니즘(Entmax, Sparsemax 등)을 비교 분석하는 프로젝트**

## 📘 Overview | 개요

This repository provides a complete implementation for comparing **advanced attention mechanisms** (such as *Entmax* and *Sparsemax*) in **Vision Transformers (ViT)**.  
It supports training, evaluation, and visualization of attention maps on **CIFAR-10**.

이 저장소는 **Vision Transformer(ViT)** 에서 **Entmax, Sparsemax 등 고급 어텐션 메커니즘**을 비교 실험하기 위한 전체 구현을 제공합니다.  
**CIFAR-10** 데이터셋을 기반으로 학습, 평가, 어텐션 맵 시각화를 지원합니다.

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

## ⚙️ Installation | 설치 방법

```bash
# Clone repository | 저장소 복제
git clone https://github.com/PoincareRice/vit_attention_comparative.git
cd vit_attention_comparative

# Create virtual environment (optional) | 가상환경 생성 (선택)
conda create -n vit-env python=3.10 -y
conda activate vit-env
#ubuntu22.04
python3.10 -m venv vit-env
source ./vit-env/bin/activate

# Install dependencies | 필요한 패키지 설치
pip install -r requirements.txt
```

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

## 🧩 Configuration | 설정 파일 예시
```yaml
model:
  base_model: "google/vit-base-patch16-224"
  attention_type: "entmax"      # options: softmax, entmax, sparsemax
  num_labels: 10

training:
  epochs: 10
  learning_rate: 3e-5
  weight_decay: 0.01
  device: "cuda"

data:
  dataset: "CIFAR10"
  image_size: 224
  batch_size: 32
  num_workers: 4

experiment:
  exp_name: "exp_001"
  checkpoint_file: "experiments/exp_001/checkpoints/vit_entmax_epoch10.pth"
```

## 🧠 Attention Mechanisms | 어텐션 메커니즘 종류
| Type | Description | 설명 |
|---|---|---|
| Softmax | Standard attention used in the original Transformer. | 기본 트랜스포머에서 사용된 표준 어텐션 방식 |
| Entmax | Introduces controlled sparsity for better interpretability. | 희소성을 부여하여 주의 집중을 더 명확히 함 |
| Sparsemax	| Produces fully sparse attention distributions. | 완전 희소한 주의 분포를 생성함 |

## 🧑‍💻 Citation | 인용 정보

If you use this project in your research, please cite:
이 프로젝트를 연구에 활용하신다면 아래와 같이 인용해 주세요.
```latex
@misc{vit_attention_comparative_2025,
  author = {Jeon, Seongyoon, Lee, Jaewon and Park, Geumrin},
  title = {A Comparative Study of Advanced Attention Mechanisms in Vision Transformers},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/<your-username>/VIT_ATTENTION_COMPARATIVE}}
}
```

## 🧩 To-Do | 향후 계획

- **Expand dataset to ImageNet**  
  Currently, CIFAR-10 is used for quick verification, but future experiments will extend to the larger and more complex ImageNet dataset to evaluate generalization performance.
  현재는 빠른 결과 확인을 위해 CIFAR-10 데이터셋을 사용했지만, 향후에는 더 복잡한 ImageNet 데이터셋으로 확장하여 모델의 일반화 성능을 검증할 예정입니다.

- **Explore additional advanced attention mechanisms**  
  Plan to implement and compare other improved attention mechanisms such as Linear Attention, Performer, and Reformer.
  Linear Attention, Performer, Reformer 등 다양한 어텐션 개선 기법을 추가로 구현하고 비교 실험할 예정입니다.

- **Implement α-Entmax parameter tuning**  
  Optimize α values to analyze the trade-off between sparsity and interpretability.
  희소성과 해석 가능성 간의 트레이드오프를 분석하기 위해 α 값을 미세 조정할 예정입니다.

- **Integrate Grad-CAM for interpretability**  
  Visualize which image regions the model attends to under different attention mechanisms.
  각 어텐션 메커니즘에서 모델이 주목하는 이미지 영역을 시각화하기 위해 Grad-CAM을 통합할 예정입니다.