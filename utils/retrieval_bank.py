from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RetrievalBank:
    z: torch.Tensor              # [N, D]
    g: torch.Tensor              # [N, D]
    y: torch.Tensor              # [N, H]
    release_ord: torch.Tensor    # [N]
    product_id: torch.Tensor     # [N]


def build_retrieval_mask(
    target_release_ord: torch.Tensor,   # [B]
    target_product_id: torch.Tensor,    # [B]
    bank_release_ord: torch.Tensor,     # [N]
    bank_product_id: torch.Tensor,      # [N]
    horizon: int,
) -> torch.Tensor:
    """
    Leakage-safe retrieval mask:
    candidate j is admissible for target i iff
    bank_release_ord[j] + horizon <= target_release_ord[i]
    and j is not the same product as i.
    """
    hist_ok = (bank_release_ord.unsqueeze(0) + horizon) <= target_release_ord.unsqueeze(1)
    not_same = bank_product_id.unsqueeze(0) != target_product_id.unsqueeze(1)
    return hist_ok & not_same


def to_device(bank: RetrievalBank, device: torch.device) -> RetrievalBank:
    return RetrievalBank(
        z=bank.z.to(device),
        g=bank.g.to(device),
        y=bank.y.to(device),
        release_ord=bank.release_ord.to(device),
        product_id=bank.product_id.to(device),
    )


def bank_as_dict(bank: RetrievalBank) -> Dict[str, torch.Tensor]:
    return {
        "z": bank.z,
        "g": bank.g,
        "y": bank.y,
        "release_ord": bank.release_ord,
        "product_id": bank.product_id,
    }