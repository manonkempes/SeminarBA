from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F


class RetrievalStore:
    def __init__(self, embeddings: torch.Tensor, sales: torch.Tensor, metadata: pd.DataFrame):
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)
        self.embeddings = F.normalize(embeddings.float().cpu(), p=2, dim=-1)
        self.sales = sales.float().cpu()
        self.metadata = metadata.reset_index(drop=True).copy()
        self.metadata['release_date'] = pd.to_datetime(self.metadata['release_date'])
        self.code_to_idx: Dict[str, int] = {
            str(code): int(idx) for idx, code in enumerate(self.metadata['external_code'].tolist())
        }
        self.split_to_indices: Dict[str, List[int]] = {
            split: self.metadata.index[self.metadata['split'] == split].tolist()
            for split in sorted(self.metadata['split'].unique())
        }
        self.release_date_ns = self.metadata['release_date'].astype('int64').to_numpy()

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def sales_horizon(self) -> int:
        return int(self.sales.shape[1])

    def lookup_indices(self, codes):
        return [self.code_to_idx[str(code)] for code in codes]

    def lookup_embeddings(self, codes):
        idxs = self.lookup_indices(codes)
        return self.embeddings[idxs]

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'embeddings': self.embeddings,
            'sales': self.sales,
            'metadata': self.metadata.to_dict(orient='list'),
        }
        torch.save(payload, path)
        self.metadata.to_csv(path.with_suffix('.metadata.csv'), index=False)

    @classmethod
    def load(cls, path: str, map_location='cpu'):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        metadata = pd.DataFrame(payload['metadata'])
        return cls(payload['embeddings'], payload['sales'], metadata)
