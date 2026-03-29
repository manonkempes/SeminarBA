from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RetrievalBank:
    z: torch.Tensor              # [N, D]
    g: torch.Tensor              # [N, D]
    y: torch.Tensor              # [N, H]
    release_ord: torch.Tensor    # [N] ordinal days
    product_id: torch.Tensor     # [N]


def build_retrieval_mask(
    target_release_ord: torch.Tensor,   # [B]
    target_product_id: torch.Tensor,    # [B]
    bank_release_ord: torch.Tensor,     # [N]
    bank_product_id: torch.Tensor,      # [N]
    horizon_weeks: int,
) -> torch.Tensor:
    """
    Leakage-safe retrieval mask:
    candidate j is admissible for target i iff

        release_date_j + horizon_weeks <= release_date_i

    release_ord is stored in ordinal DAYS, so we convert weeks -> days.
    """
    horizon_days = 7 * horizon_weeks
    hist_ok = (bank_release_ord.unsqueeze(0) + horizon_days) <= target_release_ord.unsqueeze(1)
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