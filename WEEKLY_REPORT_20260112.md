# 주간 연구 보고서 (2026-01-12)

## 전체 흐름

MemGen 프레임워크를 SmolLM3-3B 모델에 적용하여 GSM8K 수학 문제에서 27.62%p의 성능 향상을 달성하였다. MemGen(Memory Generator)은 대규모 언어 모델의 추론 과정에서 latent 벡터 형태의 메모리를 생성하고 활용하는 프레임워크로, 기존의 텍스트 기반 메모리 시스템과 달리 모델 내부에서 압축된 연속 벡터를 생성한다. 실험 결과, 기본 모델(Vanilla)의 정확도 38.79%에서 Memory Weaver 적용 시 63.16%로 24.37%p 향상되었으며, LTPO(Latent Thought Policy Optimization) 추가 적용으로 66.41%까지 도달하여 3.25%p의 추가 개선을 확인하였다. 이러한 결과는 30억 개 파라미터 규모의 소형 언어 모델에서도 latent memory 기반 추론 증강이 효과적임을 입증한다.

## 완료된 작업

### MemGen 프레임워크 검증 실험

MemGen 프레임워크는 Memory Weaver(latent 시퀀스 생성)와 Memory Trigger(메모리 삽입 시점 결정)로 구성된다. 기존의 RAG가 외부 데이터베이스에서 텍스트를 검색하여 입력 길이가 증가하는 반면, MemGen은 모델 내부에서 고정 길이의 latent 벡터를 생성하므로 추가 비용이 일정하고 별도의 인프라가 불필요하다.

### 실험 설계 및 결과 분석

본 실험은 SmolLM3-3B 모델과 GSM8K 수학 문제 데이터셋을 사용하여 진행되었다. GSM8K는 초등학교 수준의 다단계 추론이 필요한 수학 문제 모음으로, 본 실험에서는 테스트셋 전체(약 1,320개 문제)를 평가에 사용하였다. 실험 설정으로는 prompt augmentation 1회, inference augmentation 5회, latent 길이 8토큰을 적용하였다. Prompt augmentation은 입력 프롬프트 끝에 latent memory를 추가하여 초기 추론 맥락을 강화하는 기법이고, inference augmentation은 모델이 텍스트를 생성하는 중간에 latent memory를 삽입하여 추론을 보강하는 기법이다.

실험 결과, 성능 향상의 대부분은 Weaver 모듈에서 발생하였다.

### 실험 결과 요약

| 설정 | 정답 수 | 정확도 | 향상폭 | 기여도 |
|:-----|-------:|-------:|-------:|-------:|
| Vanilla (기준) | 512 / 1,320 | 38.79% | - | - |
| + Weaver | 835 / 1,322 | 63.16% | +24.37%p | 88% |
| + Weaver + LTPO | 878 / 1,322 | 66.41% | +3.25%p | 12% |
| **총 개선** | **+366** | - | **+27.62%p** | **100%** |

Vanilla 모델의 38.79% 정확도에서 Weaver 적용 시 63.16%로 향상된 것은, latent memory가 명시적인 중간 추론 단계 없이도 모델의 문제 해결 능력을 크게 향상시킴을 보여준다. 이는 기존 Chain-of-Thought(CoT)가 추론 과정을 텍스트로 명시하여 많은 토큰을 소비하는 것과 대조적이다. CoT는 해석 가능하다는 장점이 있지만 중간 단계의 오류가 누적될 수 있는 반면, latent reasoning은 벡터 공간에서 이러한 오류 전파를 완화한다.

### LTPO를 통한 추가 성능 개선

LTPO는 모델 가중치를 변경하지 않고 test-time에 latent 벡터만을 최적화하는 기법이다. 기존의 training-time optimization이 학습 데이터셋을 사용하여 모델 파라미터를 영구적으로 업데이트하는 것과 달리, LTPO는 추론 시점에 개별 입력에 대해 latent embeddings을 미세 조정한다. 최적화 과정에서 모델이 출력할 다음 토큰들의 확률 분포를 분석하여, 상위 확률 토큰들의 확신도가 높아지는 방향으로 latent를 조정한다. 본 실험에서는 학습률 0.03, 탐색 노이즈 0.1, 최적화 10회 반복의 설정을 사용하였다.

LTPO 적용 결과 Weaver 단독(63.16%)에서 66.41%로 3.25%p의 추가 향상을 달성하였다. 이는 43개의 추가 문제를 정답으로 맞춘 것에 해당한다. 전체 향상폭(27.62%p) 중 LTPO의 기여는 약 12%이지만, 이는 사전 학습된 latent도 개별 문제의 특성에 맞게 정교화될 여지가 있음을 시사한다. 특히 LTPO는 별도의 학습 데이터 없이 모델 자체의 출력 분포만으로 최적화를 수행하므로, 새로운 도메인에 즉시 적용할 수 있다는 장점이 있다.

## 결론 및 향후 계획

본 실험을 통해 MemGen 프레임워크가 소형 언어 모델에서도 효과적으로 작동함을 확인하였다. Vanilla 대비 총 27.62%p의 성능 향상 중 Weaver가 88%, LTPO가 12%를 기여하여, latent memory 생성이 핵심적인 역할을 함을 알 수 있다.

### Gnosis 기반 Correctness LTPO 연구 방향

현재 LTPO의 한계를 극복하기 위해 Gnosis 기반 correctness score를 활용한 새로운 최적화 방식을 설계하였다. 현재 LTPO는 confidence score를 reward로 사용한다. Confidence score는 모델이 다음 토큰을 예측할 때 상위 k개 토큰의 확률 평균으로 계산되는데, 이 방식은 모델이 "확신하는" 방향으로 latent를 최적화한다. 그러나 모델이 확신하지만 틀린 답을 생성하는 경우(confident-but-wrong), confidence 기반 최적화는 오히려 잘못된 방향으로 latent를 조정하게 된다.

이 문제를 해결하기 위해 Gnosis의 correctness prediction head를 LTPO에 통합하는 방안을 제안한다. Gnosis는 LLM의 hidden states와 token probabilities를 입력받아 출력의 정확성을 예측하는 경량 모듈이다. 기존의 confidence score가 토큰 확률만을 기반으로 하는 반면, Gnosis는 hidden states의 의미적 정보와 token probabilities의 패턴을 함께 분석하여 실제 정답 여부를 예측한다. Gnosis는 약 5백만 개의 파라미터만으로 수십억 파라미터 규모의 외부 평가 모델보다 우수한 정확성 예측 성능을 보인다.

제안하는 Correctness LTPO의 핵심 변경사항은 reward 함수의 교체이다. 기존 LTPO가 confidence를 최대화하는 방향으로 latent를 최적화하는 것과 달리, Correctness LTPO는 Gnosis가 예측하는 correctness probability를 최대화하는 방향으로 최적화한다. 이를 통해 "정답일 가능성이 높은" 방향으로 latent가 조정되므로, confident-but-wrong 문제를 완화할 수 있다. 또한, correctness가 특정 임계값(0.95)을 초과하면 조기 종료하여 불필요한 최적화 반복을 줄일 수 있다.

Gnosis 모듈의 학습은 GSM8K train set을 활용하여 진행할 예정이다. Weaver가 생성한 latent로 completion을 생성한 후, 정답 여부를 라벨로 사용하여 binary cross-entropy loss로 학습한다. 학습된 correctness head는 LTPO 최적화 시 reward 계산에 사용되며, Weaver GRPO 학습과 동시에 joint training하여 Weaver의 개선에 따라 correctness head도 함께 적응하도록 한다.

예상 효과로는 GSM8K 정확도 2-6%p 추가 향상(66.41% → 68-72%)과 LTPO 스텝 50% 감소(10 → 5-7 스텝)를 기대한다. 추가되는 파라미터는 약 3백만 개이며, 추론 시간 증가는 5% 이내로 예상된다.

### 기타 향후 계획

다양한 모델 크기(7B, 14B)와 다른 추론 과제(코드 생성, 논리 추론)에 대한 일반화 검증을 수행할 예정이다.

## 참고 문헌

본 연구는 다음 논문들의 방법론을 기반으로 한다. MemGen 프레임워크는 Zhang et al.(2025)이 제안한 것으로, self-evolving agent를 위한 generative latent memory 접근법을 소개하였다(arXiv:2509.24704). LTPO는 Ye et al.(2025)이 제안한 test-time reasoning 향상 기법으로, 모델 파라미터 업데이트 없이 latent 벡터만 최적화하여 추론 성능을 개선한다(arXiv:2510.04182). Gnosis는 Ghasemabadi & Niu(2025)가 제안한 LLM 자기 인식 메커니즘으로, hidden states와 attention patterns를 gated MLP로 처리하여 출력 정확성을 예측한다(arXiv:2512.20578).
