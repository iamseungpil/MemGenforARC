# Projection Layer Analysis Experiment

## 가설 (Hypothesis)

Projection-Only 모드에서 87.34%의 성능이 나오는 이유:

> **Random Query Latents → Base LLM → 다양한 Weaver Hidden States**
> 하지만 **weaver_to_reasoner Projection**을 거치면 **동일 문제끼리 Clustering**될 것이다.

즉, `weaver_to_reasoner` (학습된 MLP)가 "문제 풀기에 필요한 정보"를 추출/정렬하는 역할을 한다는 가설.

## 실험 설계

```
문제당 10개의 Random Query Latents 생성
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Reasoner Embeddings → [reasoner_to_weaver] → Weaver Space       │
│                            │                                     │
│                            ▼                                     │
│             ★ Point 1: After reasoner_to_weaver                  │
│                            │                                     │
│                            ▼                                     │
│              concat with Random Query Latents                    │
│                            │                                     │
│                            ▼                                     │
│              Base LLM Forward (LoRA 비활성화)                     │
│                            │                                     │
│                            ▼                                     │
│             ★ Point 2: Weaver Hidden States                      │
│                  (weaver_to_reasoner 전)                         │
│                            │                                     │
│                            ▼                                     │
│                  [weaver_to_reasoner]                            │
│                            │                                     │
│                            ▼                                     │
│             ★ Point 3: Memory Tokens                             │
│                  (weaver_to_reasoner 후)                         │
└─────────────────────────────────────────────────────────────────┘
```

## 예상 결과 (가설이 맞다면)

| 시각화 포인트 | 예상 패턴 |
|--------------|----------|
| Point 1 (r2w 후) | 문제별 클러스터 (입력 정보 유지) |
| Point 2 (LLM 후) | **분산됨** (Random query latents로 인해) |
| Point 3 (w2r 후) | **문제별 클러스터!** (weaver_to_reasoner가 정렬) |

## 사용법

### 1. 분석 실행

```bash
cd /home/jovyan/MemGenforARC/experiments/projection_analysis

# Config 파일 사용
python run_analysis.py --config config.yaml

# 또는 커맨드라인 인자 사용
python run_analysis.py \
    --num_problems 20 \
    --num_random_latents 10 \
    --model_name Qwen/Qwen3-8B \
    --load_weaver_path /path/to/trained/weaver \
    --output_dir ./results/my_experiment
```

### 2. 시각화

```bash
# 모든 방법으로 시각화 (t-SNE, UMAP, PCA)
python visualize.py --input_dir ./results/20260116_123456

# 특정 방법만
python visualize.py --input_dir ./results/20260116_123456 --method tsne
```

## 출력 파일

```
results/20260116_123456/
├── embeddings.pt           # 수집된 임베딩 데이터
├── problems.pt             # 분석한 문제 텍스트
├── metrics.txt             # 클러스터링 메트릭
└── visualizations/
    ├── point1_tsne.png     # Point 1 t-SNE
    ├── point2_tsne.png     # Point 2 t-SNE
    ├── point3_tsne.png     # Point 3 t-SNE
    ├── comparison_tsne.png # 3개 포인트 비교
    ├── point1_umap.png     # Point 1 UMAP
    ├── ...
    └── comparison_pca.png  # PCA 비교
```

## 클러스터링 메트릭

- **Silhouette Score**: -1 ~ 1, 높을수록 클러스터 품질 좋음
- **Calinski-Harabasz**: 높을수록 클러스터 품질 좋음
- **Davies-Bouldin**: 낮을수록 클러스터 품질 좋음

## 추가 실험 아이디어

### 1. Trained vs Random Projections 비교

```bash
# Trained projections
python run_analysis.py \
    --load_weaver_path /path/to/trained/weaver \
    --output_dir ./results/trained

# Random projections (load_weaver_path 없이)
python run_analysis.py \
    --output_dir ./results/random
```

### 2. weaver_to_reasoner만 학습 vs 둘 다 학습

코드 수정 필요 (향후 확장)

## 의존성

```bash
pip install scikit-learn matplotlib seaborn
pip install umap-learn  # UMAP 사용 시
```

## 메인 코드 영향

**없음** - 이 실험은 메인 코드를 import만 하고 수정하지 않습니다.
