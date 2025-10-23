"""
Contrastive Learning Head with Temperature Scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """
    - L2 normalize 후 dot-product(sim) 계산
    - learnable temperature(τ)로 로짓 스케일 조정
    """
    def __init__(self, init_tau: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            # τ>0 보장 위해 log-parameterize
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))
        else:
            self.register_buffer("tau", torch.tensor(init_tau))
        self.learnable = learnable

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: Query embeddings [B, D]
            k: Key embeddings [B, D]

        Returns:
            logits: Similarity matrix scaled by temperature [B, B]
        """
        # L2 정규화
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Dot product similarity
        logits = q @ k.T  # [B, B]

        # Temperature scaling
        tau = torch.exp(self.log_tau) if self.learnable else self.tau
        logits = logits / tau.clamp_min(1e-6)

        return logits


@torch.no_grad()
def diag_consistency_report(q: torch.Tensor, k: torch.Tensor, log_prefix: str = "[diag]"):
    """
    대각선 정합성 즉시 점검. z_gap이 확실히 양수/0.6↑면 OK.

    Args:
        q: Query embeddings [B, D]
        k: Key embeddings [B, D]
        log_prefix: 로그 접두사

    Returns:
        z_gap: Standardized similarity gap
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    sim = q @ k.T

    B = sim.size(0)
    diag = sim.diag()
    offdiag = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)]

    # Z-score gap 계산
    gap = (diag.mean() - offdiag.mean()) / (offdiag.std(unbiased=False) + 1e-6)

    print(f"{log_prefix} B={B} diag_mean={diag.mean().item():.4f} "
          f"off_mean={offdiag.mean().item():.4f} off_std={offdiag.std(unbiased=False).item():.4f} "
          f"z_gap={gap:.3f} min={sim.min().item():.4f} max={sim.max().item():.4f}")

    return gap
