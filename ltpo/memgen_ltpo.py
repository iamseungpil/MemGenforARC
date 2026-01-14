"""
MemGen LTPO Optimizer

LTPO 원본(LTPO_backup/ltpo.py)을 MemGen에 맞게 수정.
- get_confidence(): 원본 그대로
- optimize(): 원본 최적화 루프를 MemGen 구조에 맞게 적용
"""

import torch
from .config import LTPOConfig


class MemGenLTPOOptimizer:
    """MemGen용 LTPO 최적화기"""

    def __init__(self, config: LTPOConfig):
        self.lr = config.lr
        self.sigma = config.sigma
        self.sigma_decay = config.sigma_decay
        self.max_steps = config.max_steps
        self.reward_threshold = config.reward_threshold
        self.top_k = config.top_k
        self.use_auto_grad = config.use_auto_grad
        self.disable_best_reward = config.disable_best_reward
        self.verbose = config.verbose

    def get_confidence(
        self,
        model,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_start_idx: int,
        latent_end_idx: int,
    ) -> torch.Tensor:
        """
        Confidence 계산 - LTPO 원본(ltpo.py:102-117) 기반

        차이점:
        - 원본: range(start, end+1) - gen_prompt 포함
        - MemGen: range(start, end) - latent만 (gen_prompt 없음)
        """
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

        confidence = 0.0
        for idx in range(latent_start_idx, latent_end_idx):
            topk = torch.topk(probs[idx], k=self.top_k, largest=True)[0]
            confidence -= torch.sum(torch.log(topk + 1e-10)) / self.top_k

        num_tokens = latent_end_idx - latent_start_idx
        return confidence / num_tokens

    def optimize(
        self,
        model,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_start_idx: int,
        latent_end_idx: int,
    ) -> torch.Tensor:
        """
        Latent 최적화 - LTPO 원본(ltpo.py:157-224) 기반

        Args:
            model: Reasoner (confidence 계산용)
            inputs_embeds: [1, seq_len, hidden_dim]
            attention_mask: [1, seq_len]
            latent_start_idx: latent 시작 위치
            latent_end_idx: latent 끝 위치 (exclusive)

        Returns:
            optimized_inputs_embeds: 최적화된 embedding
        """
        device = inputs_embeds.device
        sigma = self.sigma

        # latent 추출 - 원본 ltpo.py:157-163
        latent = inputs_embeds[0, latent_start_idx:latent_end_idx].clone()

        if self.use_auto_grad:
            latent = torch.nn.Parameter(latent.detach().requires_grad_(True))
            optimizer = torch.optim.Adam([latent], lr=self.lr, maximize=True)

        best_reward = 0.0
        best_reward_step = 0
        best_latent = latent.clone().detach()

        # 최적화 루프 - 원본 ltpo.py:168-218
        for i in range(self.max_steps):
            if self.use_auto_grad:
                optimizer.zero_grad()

            # noise 추가
            epsilon = torch.normal(mean=0.0, std=sigma, size=latent.shape).to(device)
            latent_cand = latent + epsilon

            # inputs_embeds에 candidate 삽입
            inputs_cand = inputs_embeds.clone()
            inputs_cand[0, latent_start_idx:latent_end_idx] = latent_cand

            # confidence 계산
            if self.use_auto_grad:
                reward = self.get_confidence(
                    model=model,
                    inputs_embeds=inputs_cand,
                    attention_mask=attention_mask,
                    latent_start_idx=latent_start_idx,
                    latent_end_idx=latent_end_idx,
                )
                reward.backward(retain_graph=True)
                optimizer.step()
            else:
                with torch.no_grad():
                    reward = self.get_confidence(
                        model=model,
                        inputs_embeds=inputs_cand,
                        attention_mask=attention_mask,
                        latent_start_idx=latent_start_idx,
                        latent_end_idx=latent_end_idx,
                    )
                # REINFORCE style - 원본 ltpo.py:203-204
                grad_ascent = self.lr * reward * epsilon / (sigma ** 2)
                latent = latent + grad_ascent

            sigma *= self.sigma_decay

            if self.verbose:
                print(f'[LTPO] Step {i}: reward = {reward:.4f}')

            del epsilon, latent_cand, inputs_cand
            torch.cuda.empty_cache()

            # best 저장 - 원본 ltpo.py:213-216
            reward_val = reward.item() if torch.is_tensor(reward) else reward
            if reward_val > best_reward:
                best_reward = reward_val
                best_reward_step = i
                best_latent = latent.clone().detach() if self.use_auto_grad else latent.clone()

            if self.reward_threshold > 0 and reward_val >= self.reward_threshold:
                break

        if self.verbose:
            print(f'[LTPO] Best: reward={best_reward:.4f}, step={best_reward_step}')

        # 결과 반환 - 원본 ltpo.py:220-224
        result = inputs_embeds.clone()
        if self.disable_best_reward:
            result[0, latent_start_idx:latent_end_idx] = latent.detach() if self.use_auto_grad else latent
        else:
            result[0, latent_start_idx:latent_end_idx] = best_latent

        return result
