import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalModule(nn.Module):
    """
    Retrieval module for explicit historical analog retrieval.

    Input:
        z_i:       [B, D]   multimodal product embedding
        g_i:       [B, D]   pooled trend embedding
        bank_z:    [N, D]   product embeddings in memory bank
        bank_g:    [N, D]   pooled trend embeddings in memory bank
        bank_y:    [N, H]   observed sales trajectories in memory bank
        valid_mask:[B, N]   admissibility mask for leakage prevention

    Output:
        z_tilde:   [B, D]   retrieval-augmented product embedding
        aux:       dict     debugging info
    """
    def __init__(
        self,
        prod_dim: int,
        trend_dim: int,
        horizon: int,
        retrieval_dim: int = 64,
        topk: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.topk = topk
        self.horizon = horizon
        self.retrieval_dim = retrieval_dim

        self.query_proj = nn.Linear(prod_dim + trend_dim, retrieval_dim)
        self.key_proj = nn.Linear(prod_dim + trend_dim, retrieval_dim)

        self.sales_projector = nn.Sequential(
            nn.Linear(horizon, retrieval_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(retrieval_dim, retrieval_dim),
        )

        self.compatibility = nn.Sequential(
            nn.Linear(retrieval_dim * 2, retrieval_dim),
            nn.Tanh(),
            nn.Linear(retrieval_dim, 1),
        )

        self.augment = nn.Sequential(
            nn.Linear(prod_dim + retrieval_dim, prod_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_i: torch.Tensor,
        g_i: torch.Tensor,
        bank_z: torch.Tensor,
        bank_g: torch.Tensor,
        bank_y: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        # Query/key representations
        q = self.query_proj(torch.cat([z_i, g_i], dim=-1))          # [B, R]
        k = self.key_proj(torch.cat([bank_z, bank_g], dim=-1))      # [N, R]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        sim = q @ k.transpose(0, 1)                                 # [B, N]
        sim = sim.masked_fill(~valid_mask, -1e9)

        k_eff = min(self.topk, sim.size(1))
        top_sim, top_idx = torch.topk(sim, k=k_eff, dim=1)          # [B, K]

        retrieved_y = bank_y[top_idx]                               # [B, K, H]
        proj_y = self.sales_projector(retrieved_y)                  # [B, K, R]

        q_rep = q.unsqueeze(1).expand(-1, k_eff, -1)                # [B, K, R]
        alpha_logits = self.compatibility(torch.cat([q_rep, proj_y], dim=-1)).squeeze(-1)
        alpha_logits = alpha_logits.masked_fill(top_sim < -1e8, -1e9)
        alpha = torch.softmax(alpha_logits, dim=1)                  # [B, K]

        r_i = torch.sum(alpha.unsqueeze(-1) * proj_y, dim=1)        # [B, R]
        z_tilde = self.augment(torch.cat([z_i, r_i], dim=-1))       # [B, D]

        aux = {
            "top_idx": top_idx,
            "top_sim": top_sim,
            "alpha": alpha,
            "retrieved_y": retrieved_y,
        }
        return z_tilde, aux