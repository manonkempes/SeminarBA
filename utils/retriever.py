from typing import Dict, List

import pandas as pd
import torch

from utils.embedding_store import RetrievalStore


class TemporalNearestNeighborRetriever:
    def __init__(self, store: RetrievalStore, retrieval_k: int = 5, retrieval_observation_horizon: int = 12):
        self.store = store
        self.retrieval_k = retrieval_k
        self.retrieval_observation_horizon = retrieval_observation_horizon
        self.horizon_delta_ns = pd.Timedelta(weeks=retrieval_observation_horizon).value
        self.device = torch.device('cpu')
        self.embeddings = store.embeddings
        self.sales = store.sales
        self.release_date_ns = torch.as_tensor(store.release_date_ns, dtype=torch.long)
        self.bank_indices: Dict[str, torch.Tensor] = {}
        self.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.embeddings = self.store.embeddings.to(self.device)
        self.sales = self.store.sales.to(self.device)
        self.release_date_ns = torch.as_tensor(self.store.release_date_ns, dtype=torch.long, device=self.device)
        self.bank_indices = {
            split: torch.tensor(indices, dtype=torch.long, device=self.device)
            for split, indices in self.store.split_to_indices.items()
        }
        return self

    def _candidate_bank_splits(self, target_split: str) -> List[str]:
        if target_split in {'subtrain', 'val'}:
            return ['subtrain']
        if target_split == 'test':
            return ['subtrain', 'val']
        raise ValueError(f'Unsupported target split: {target_split}')

    def _candidate_indices(self, target_idx: int, target_split: str) -> torch.Tensor:
        bank_parts = [self.bank_indices[split] for split in self._candidate_bank_splits(target_split) if split in self.bank_indices]
        if not bank_parts:
            return torch.empty(0, dtype=torch.long, device=self.device)
        candidates = torch.cat(bank_parts)
        candidates = candidates[candidates != int(target_idx)]
        if candidates.numel() == 0:
            return candidates
        target_release = self.release_date_ns[int(target_idx)]
        admissible = self.release_date_ns[candidates] + self.horizon_delta_ns <= target_release
        return candidates[admissible]

    def lookup_target_embeddings(self, target_codes, device=None) -> torch.Tensor:
        if device is not None and torch.device(device) != self.device:
            self.to(device)
        idxs = self.store.lookup_indices(target_codes)
        return self.embeddings[idxs]

    def retrieve_batch(self, target_codes, target_split: str, device=None):
        if device is not None and torch.device(device) != self.device:
            self.to(device)

        target_codes = [str(code) for code in target_codes]
        target_indices = self.store.lookup_indices(target_codes)
        target_embeddings = self.embeddings[target_indices]
        value_horizon = min(self.retrieval_observation_horizon, self.sales.shape[1])

        batch_sales = []
        batch_scores = []
        batch_weights = []
        batch_availability = []
        details = []

        for batch_idx, target_idx in enumerate(target_indices):
            candidate_indices = self._candidate_indices(target_idx=target_idx, target_split=target_split)
            target_embedding = target_embeddings[batch_idx]

            if candidate_indices.numel() == 0:
                batch_sales.append(torch.zeros(self.retrieval_k, value_horizon, device=self.device))
                batch_scores.append(torch.full((self.retrieval_k,), float('-inf'), device=self.device))
                batch_weights.append(torch.zeros(self.retrieval_k, device=self.device))
                batch_availability.append(torch.tensor([0.0], device=self.device))
                details.append({
                    'target_code': target_codes[batch_idx],
                    'target_idx': int(target_idx),
                    'target_release_date': str(self.store.metadata.iloc[target_idx]['release_date']),
                    'admissible_candidate_count': 0,
                    'neighbor_codes': [],
                    'neighbor_release_dates': [],
                    'cosine_scores': [],
                    'softmax_weights': [],
                    'retrieved_sales': [],
                    'retrieval_available': 0,
                })
                continue

            similarities = torch.mv(self.embeddings[candidate_indices], target_embedding)
            k = min(self.retrieval_k, int(candidate_indices.numel()))
            top_scores, top_pos = torch.topk(similarities, k=k, largest=True, sorted=True)
            top_indices = candidate_indices[top_pos]
            top_sales = self.sales[top_indices, :value_horizon]
            top_weights = torch.softmax(top_scores, dim=0)

            if k < self.retrieval_k:
                padded_sales = torch.zeros(self.retrieval_k, value_horizon, device=self.device)
                padded_scores = torch.full((self.retrieval_k,), float('-inf'), device=self.device)
                padded_weights = torch.zeros(self.retrieval_k, device=self.device)
                padded_sales[:k] = top_sales
                padded_scores[:k] = top_scores
                padded_weights[:k] = top_weights
            else:
                padded_sales = top_sales
                padded_scores = top_scores
                padded_weights = top_weights

            top_metadata = self.store.metadata.iloc[top_indices.detach().cpu().tolist()]
            details.append({
                'target_code': target_codes[batch_idx],
                'target_idx': int(target_idx),
                'target_release_date': str(self.store.metadata.iloc[target_idx]['release_date']),
                'admissible_candidate_count': int(candidate_indices.numel()),
                'neighbor_codes': top_metadata['external_code'].tolist(),
                'neighbor_release_dates': [str(x) for x in top_metadata['release_date'].tolist()],
                'cosine_scores': top_scores.detach().cpu().tolist(),
                'softmax_weights': top_weights.detach().cpu().tolist(),
                'retrieved_sales': top_sales.detach().cpu().tolist(),
                'retrieval_available': 1,
            })

            batch_sales.append(padded_sales)
            batch_scores.append(padded_scores)
            batch_weights.append(padded_weights)
            batch_availability.append(torch.tensor([1.0], device=self.device))

        return {
            'target_embeddings': target_embeddings,
            'neighbor_sales': torch.stack(batch_sales, dim=0),
            'cosine_scores': torch.stack(batch_scores, dim=0),
            'weights': torch.stack(batch_weights, dim=0),
            'availability': torch.stack(batch_availability, dim=0),
            'details': details,
        }
