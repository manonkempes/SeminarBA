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

def build_retrieval_bank(model, loader, device):
    model.eval()

    z_list = []
    g_list = []
    y_list = []
    release_ord_list = []
    product_id_list = []

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]

            (
                item_sales,
                category,
                color,
                fabric,
                temporal_features,
                gtrends,
                images,
                release_ord,
                product_id,
            ) = batch

            z = model.encode_static(
                category,
                color,
                fabric,
                temporal_features,
                images,
            )
            _, g = model.encode_trends(gtrends)

            z_list.append(z.detach().cpu())
            g_list.append(g.detach().cpu())
            y_list.append(item_sales.detach().cpu())
            release_ord_list.append(release_ord.detach().cpu().view(-1))
            product_id_list.append(product_id.detach().cpu().view(-1))

    return RetrievalBank(
        z=torch.cat(z_list, dim=0),
        g=torch.cat(g_list, dim=0),
        y=torch.cat(y_list, dim=0),
        release_ord=torch.cat(release_ord_list, dim=0),
        product_id=torch.cat(product_id_list, dim=0),
    )