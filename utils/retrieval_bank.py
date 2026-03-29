from dataclasses import dataclass
import torch


@dataclass
class RetrievalBank:
    z: torch.Tensor
    g: torch.Tensor
    y: torch.Tensor
    release_ord: torch.Tensor
    product_id: torch.Tensor


def build_retrieval_mask(
    target_release_ord: torch.Tensor,
    target_product_id: torch.Tensor,
    bank_release_ord: torch.Tensor,
    bank_product_id: torch.Tensor,
    horizon_weeks: int,
) -> torch.Tensor:
    """
    release_ord is in DAYS because it comes from Timestamp.toordinal().
    The forecast horizon is in WEEKS, so we convert it to days.
    Candidate j is admissible for target i iff:
        bank_release_ord[j] + 7 * horizon_weeks <= target_release_ord[i]
    and it is not the same product.
    """
    horizon_days = horizon_weeks * 7

    hist_ok = (bank_release_ord.unsqueeze(0) + horizon_days) <= target_release_ord.unsqueeze(1)
    not_same = bank_product_id.unsqueeze(0) != target_product_id.unsqueeze(1)

    return hist_ok & not_same