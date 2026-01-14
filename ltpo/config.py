"""
LTPO Config - LTPO 원본(LTPO_backup/main.py) 파라미터 그대로 유지
"""

from dataclasses import dataclass


@dataclass
class LTPOConfig:
    """LTPO 설정"""
    enabled: bool = False
    lr: float = 0.03
    sigma: float = 0.1
    sigma_decay: float = 0.99
    max_steps: int = 10
    reward_threshold: float = -1  # -1 = disabled
    top_k: int = 10
    use_auto_grad: bool = True
    disable_best_reward: bool = False
    verbose: int = 1
