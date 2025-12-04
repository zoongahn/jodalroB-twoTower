import torch
import torch.nn as nn

class FeatureProjector(nn.Module):
    def __init__(self, num_dim: int, text_dim: int, num_proj_dim: int = 128, text_proj_dim: int = 128):
        super().__init__()
        # 수치형 projection: (29 → 128)
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, num_proj_dim),
            nn.ReLU(),
            nn.Linear(num_proj_dim, num_proj_dim),
        )
        # 텍스트 projection: (768 → 128)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.ReLU(),
            nn.Linear(text_proj_dim, text_proj_dim),
        )

    def forward(self, dense: torch.Tensor, text_dict: dict[str, torch.Tensor]):
        # dense: (B, num_dim=29)
        # 🔧 입력 clipping (극단적인 값 방지)
        if dense is not None:
            dense = torch.clamp(dense, min=-1e4, max=1e4)

        dense_proj = self.num_proj(dense)  # (B, 128)

        # 🔧 출력 clipping (Inf 방지)
        dense_proj = torch.clamp(dense_proj, min=-1e4, max=1e4)

        # text_dict: {"colname": (B, 768), ...}
        text_proj = {}
        for col, x in text_dict.items():
            x_clamped = torch.clamp(x, min=-1e4, max=1e4)
            proj = self.text_proj(x_clamped)
            text_proj[col] = torch.clamp(proj, min=-1e4, max=1e4)

        return dense_proj, text_proj