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
        dense_proj = self.num_proj(dense)  # (B, 128)

        # text_dict: {"colname": (B, 768), ...}
        text_proj = {
            col: self.text_proj(x) for col, x in text_dict.items()
        }  # {"colname": (B, 128)}

        return dense_proj, text_proj