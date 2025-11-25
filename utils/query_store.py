import random
from collections import defaultdict, OrderedDict
from typing import Iterable, List, Optional, Tuple, Dict, Union

import torch
import torch.nn.functional as F


class ConstraintStore:
    """Keeps track of must-link / cannot-link relations across the dataset."""

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self._relations = defaultdict(dict)
        self._pair_index = OrderedDict()
        self._parent = list(range(num_samples))
        self._rank = [0] * num_samples
        self._component_members = {i: {i} for i in range(num_samples)}
        self._cannot_relations = defaultdict(set)

    def add(self, a: int, b: int, is_same: bool) -> None:
        if a == b:
            return
        if is_same:
            self._add_must_link(int(a), int(b))
        else:
            self._add_cannot_link(int(a), int(b))

    def _find_root(self, idx: int) -> int:
        parent = self._parent
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def _record_pair(self, a: int, b: int, value: int) -> None:
        if a == b:
            return
        key = tuple(sorted((a, b)))
        existing = self._pair_index.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"Conflicting constraints for pair {key}: {existing} vs {value}")
        self._pair_index[key] = value
        self._relations[a][b] = value
        self._relations[b][a] = value

    def _apply_pair_values(self, group_a, group_b, value: int) -> None:
        for i in group_a:
            for j in group_b:
                if i == j:
                    continue
                self._record_pair(int(i), int(j), value)

    def _union_roots(self, root_a: int, root_b: int) -> int:
        if root_a == root_b:
            return root_a
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1
        members_b = self._component_members.pop(root_b, {root_b})
        self._component_members[root_a].update(members_b)

        cannot_a = self._cannot_relations.get(root_a, set())
        cannot_b = self._cannot_relations.pop(root_b, set())
        merged = {self._find_root(c) for c in (cannot_a | cannot_b)}
        merged.discard(root_a)
        self._cannot_relations[root_a] = merged
        for other in merged:
            rel_set = self._cannot_relations.setdefault(other, set())
            if root_b in rel_set:
                rel_set.remove(root_b)
            rel_set.add(root_a)
        return root_a

    def _add_must_link(self, a: int, b: int) -> None:
        root_a = self._find_root(a)
        root_b = self._find_root(b)
        if root_a == root_b:
            self._record_pair(a, b, 1)
            return
        if root_b in self._cannot_relations.get(root_a, set()):
            raise ValueError(f"Cannot must-link nodes from conflicting components: {a}, {b}")
        self._record_pair(a, b, 1)
        members_a = list(self._component_members[root_a])
        members_b = list(self._component_members[root_b])
        self._apply_pair_values(members_a, members_b, 1)
        new_root = self._union_roots(root_a, root_b)
        for other in list(self._cannot_relations.get(new_root, set())):
            other_root = self._find_root(other)
            members_other = list(self._component_members.get(other_root, []))
            if members_other:
                self._apply_pair_values(self._component_members[new_root], members_other, -1)

    def _add_cannot_link(self, a: int, b: int) -> None:
        root_a = self._find_root(a)
        root_b = self._find_root(b)
        if root_a == root_b:
            raise ValueError(f"Cannot-link within same component: {a}, {b}")
        self._record_pair(a, b, -1)
        members_a = self._component_members[root_a]
        members_b = self._component_members[root_b]
        self._apply_pair_values(members_a, members_b, -1)
        self._cannot_relations[root_a].add(root_b)
        self._cannot_relations[root_b].add(root_a)

    def has_relation(self, a: int, b: int) -> bool:
        return b in self._relations.get(a, {})

    def neighbors(self, a: int) -> Dict[int, int]:
        """Returns known relations for sample a."""
        return self._relations.get(int(a), {})

    def iter_pairs(self):
        return self._pair_index.items()

    def build_batch_matrix(
        self, indices: Iterable[int], device: torch.device
    ) -> torch.Tensor:
        idx_list = [int(i) for i in indices]
        n = len(idx_list)
        matrix = torch.zeros(n, n, device=device)
        index_map = {idx: pos for pos, idx in enumerate(idx_list)}
        for idx, pos in index_map.items():
            neighbors = self._relations.get(idx)
            if not neighbors:
                continue
            for nb, value in neighbors.items():
                local = index_map.get(nb)
                if local is None:
                    continue
                matrix[pos, local] = value
        return matrix

    def sample_pairs(
        self,
        max_pairs: int,
        ready_mask: Optional[torch.Tensor] = None,
        shuffle: bool = True,
    ) -> List[Tuple[int, int, int]]:
        if not self._pair_index:
            return []
        items = list(self._pair_index.items())
        if shuffle:
            random.shuffle(items)
        sampled: List[Tuple[int, int, int]] = []
        for (a, b), value in items:
            if ready_mask is not None:
                if not (ready_mask[a] and ready_mask[b]):
                    continue
            sampled.append((a, b, value))
            if len(sampled) >= max_pairs:
                break
        return sampled

    def __len__(self) -> int:
        return len(self._pair_index)


class MemoryBank:
    """Momentum memory bank for storing representations per sample index."""

    def __init__(
        self, num_samples: int, dim: int, device: torch.device, momentum: float = 0.0
    ):
        self.device = device
        self.bank = torch.zeros(num_samples, dim, device=device)
        self.mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
        self.momentum = momentum

    def update(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        if values.device != self.device:
            values = values.to(self.device)
        indices = indices.to(self.device)
        if self.momentum > 0:
            existing = self.bank.index_select(0, indices)
            updated = existing * self.momentum + (1 - self.momentum) * values
            self.bank.index_copy_(0, indices, updated)
        else:
            self.bank.index_copy_(0, indices, values)
        self.mask.index_fill_(0, indices, True)

    @property
    def ready_mask(self) -> torch.Tensor:
        return self.mask

    def ready_indices(self) -> torch.Tensor:
        return torch.nonzero(self.mask, as_tuple=False).squeeze(-1)


class GlobalQuerySelector:
    """Proposes informative global pairs using the memory banks."""

    def __init__(
        self,
        constraint_store: ConstraintStore,
        feature_bank: MemoryBank,
        cluster_bank: MemoryBank,
        candidate_pool: int = 2048,
        sup_epsilon: float = 0.1,
        minmax_eps: float = 1e-6,
        total_queries: int = 1,
    ):
        self.store = constraint_store
        self.feature_bank = feature_bank
        self.cluster_bank = cluster_bank
        self.candidate_pool = candidate_pool
        self.sup_epsilon = float(sup_epsilon)
        self.minmax_eps = float(minmax_eps)
        self.total_queries = max(1, int(total_queries))
        self.device = feature_bank.bank.device

    def ready(self) -> bool:
        joint_mask = self.feature_bank.ready_mask & self.cluster_bank.ready_mask
        return int(joint_mask.sum().item()) >= 2

    def _min_max_norm(
        self, tensor: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if valid_mask is None:
            values = tensor.view(-1)
        else:
            values = tensor[valid_mask]
        if values.numel() == 0:
            return torch.zeros_like(tensor)
        min_val = values.min()
        max_val = values.max()
        if torch.isclose(max_val, min_val):
            return torch.zeros_like(tensor)
        normed = (tensor - min_val) / (max_val - min_val + 1e-12)
        normed = torch.where(valid_mask, normed, torch.zeros_like(normed)) if valid_mask is not None else normed
        return normed

    def _build_neighbor_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        dim = self.feature_bank.bank.size(1)
        agg = torch.zeros(indices.size(0), dim, device=self.device)
        ready_mask = self.feature_bank.ready_mask
        for local_idx, sample_idx in enumerate(indices.tolist()):
            neighbors = self.store.neighbors(sample_idx)
            if not neighbors:
                continue
            pos_ids, neg_ids = [], []
            for nb, value in neighbors.items():
                if not bool(ready_mask[int(nb)].item()):
                    continue
                if value == 1:
                    pos_ids.append(nb)
                elif value == -1:
                    neg_ids.append(nb)
            if not pos_ids and not neg_ids:
                continue
            vec = torch.zeros(dim, device=self.device)
            if pos_ids:
                pos_tensor = torch.tensor(pos_ids, device=self.device, dtype=torch.long)
                vec = vec + self.feature_bank.bank.index_select(0, pos_tensor).sum(dim=0)
            if neg_ids:
                neg_tensor = torch.tensor(neg_ids, device=self.device, dtype=torch.long)
                vec = vec - self.feature_bank.bank.index_select(0, neg_tensor).sum(dim=0)
            agg[local_idx] = vec
        norm = agg.norm(dim=1, keepdim=True)
        norm_flat = norm.squeeze(-1)
        nonzero = norm_flat > 0
        if nonzero.any():
            agg[nonzero] = agg[nonzero] / norm_flat[nonzero].unsqueeze(-1).clamp_min(1e-12)
        agg[~nonzero] = 0
        return agg

    def propose_pair(self, remaining_queries: Optional[int] = None) -> Optional[Tuple[int, int]]:
        joint_mask = self.feature_bank.ready_mask & self.cluster_bank.ready_mask
        ready_indices = torch.nonzero(joint_mask, as_tuple=False).squeeze(-1)
        if ready_indices.numel() < 2:
            return None

        if ready_indices.numel() > self.candidate_pool:
            perm = torch.randperm(ready_indices.numel(), device=ready_indices.device)[
                : self.candidate_pool
            ]
            ready_indices = ready_indices[perm]

        feats = self.feature_bank.bank.index_select(0, ready_indices)
        feats = F.normalize(feats, dim=1, eps=1e-12)
        sim_matrix = torch.matmul(feats, feats.t()).clamp(-1.0, 1.0)
        num = ready_indices.size(0)
        valid_mask = ~torch.eye(num, dtype=torch.bool, device=self.device)

        sup_raw = -torch.abs(sim_matrix - self.sup_epsilon)
        sup_scores = self._min_max_norm(sup_raw, valid_mask)

        neighbor_vectors = self._build_neighbor_vectors(ready_indices)
        q_forward = torch.matmul(feats, neighbor_vectors.t())
        q_forward = torch.nan_to_num(q_forward, nan=0.0)
        q_raw = 0.5 * (q_forward + q_forward.t())

        p_probs = torch.clamp(self._min_max_norm(sim_matrix, valid_mask), min=self.minmax_eps, max=1.0)
        q_probs = torch.clamp(self._min_max_norm(q_raw, valid_mask), min=self.minmax_eps, max=1.0)

        kl_raw = torch.zeros_like(sim_matrix)
        kl_raw[valid_mask] = p_probs[valid_mask] * torch.log(
            p_probs[valid_mask] / q_probs[valid_mask]
        )
        shp_scores = self._min_max_norm(kl_raw, valid_mask)

        remaining = remaining_queries if remaining_queries is not None else 0.5 * self.total_queries
        ratio = float(max(0.0, min(1.0, remaining / self.total_queries)))
        joint_scores = ratio * sup_scores + (1 - ratio) * shp_scores
        joint_scores = joint_scores.masked_fill(~valid_mask, float("-inf"))

        ready_list = ready_indices.tolist()
        index_lookup = {int(idx): pos for pos, idx in enumerate(ready_list)}
        for (a, b), _ in self.store.iter_pairs():
            pa = index_lookup.get(a)
            pb = index_lookup.get(b)
            if pa is None or pb is None:
                continue
            joint_scores[pa, pb] = float("-inf")
            joint_scores[pb, pa] = float("-inf")

        max_score = torch.max(joint_scores)
        if not torch.isfinite(max_score):
            return None
        flat_idx = torch.argmax(joint_scores)
        row = int(flat_idx // num)
        col = int(flat_idx % num)
        anchor = int(ready_indices[row].item())
        candidate = int(ready_indices[col].item())
        if anchor == candidate:
            return None
        return anchor, candidate


DebugRecord = Dict[str, Union[int, float]]


class EpochQuerySelector:
    """Selects informative sample-center pairs once per epoch."""

    def __init__(
        self,
        constraint_store: ConstraintStore,
        feature_bank: MemoryBank,
        cluster_bank: MemoryBank,
        num_clusters: int,
        queries_per_epoch: int,
        candidate_pool: int = 2048,
        neighbor_k: int = 20,
        kmeans_iters: int = 10,
    ):
        self.store = constraint_store
        self.feature_bank = feature_bank
        self.cluster_bank = cluster_bank
        self.num_clusters = max(1, int(num_clusters))
        self.queries_per_epoch = max(1, int(queries_per_epoch))
        self.candidate_pool = candidate_pool
        self.neighbor_k = max(1, int(neighbor_k))
        self.kmeans_iters = max(1, int(kmeans_iters))
        self.device = feature_bank.bank.device

    def ready(self) -> bool:
        joint_mask = self.feature_bank.ready_mask & self.cluster_bank.ready_mask
        return int(joint_mask.sum().item()) >= 2

    def propose_epoch_pairs(self, max_pairs: int) -> Tuple[List[Tuple[int, int]], List[DebugRecord]]:
        if not self.ready():
            return [], []
        limit = min(self.queries_per_epoch, max_pairs)
        if limit <= 0:
            return [], []
        gathered = self._gather_candidates()
        if gathered is None:
            return [], []
        candidate_indices, feats = gathered
        centers, assignments, center_distances = self._run_kmeans(feats)
        hardness = self._compute_hardness(feats, centers)
        representativeness = self._compute_representativeness(feats)
        selected_positions = self._select_samples(
            hardness, representativeness, feats, limit
        )
        cluster_members = self._build_cluster_members(assignments, center_distances)
        candidate_list = candidate_indices.tolist()
        pairs: List[Tuple[int, int]] = []
        debug: List[DebugRecord] = []
        for pos in selected_positions:
            cluster_id = int(assignments[pos].item())
            partner_pos = self._pick_cluster_partner(cluster_members, cluster_id, pos)
            if partner_pos is None:
                continue
            anchor_idx = int(candidate_list[pos])
            partner_idx = int(candidate_list[partner_pos])
            if anchor_idx == partner_idx:
                continue
            if self.store.has_relation(anchor_idx, partner_idx):
                continue
            pairs.append((anchor_idx, partner_idx))
            debug.append(
                {
                    "anchor": anchor_idx,
                    "partner": partner_idx,
                    "hardness": float(hardness[pos].item()),
                }
            )
        return pairs, debug

    def _gather_candidates(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        joint_mask = self.feature_bank.ready_mask & self.cluster_bank.ready_mask
        ready_indices = torch.nonzero(joint_mask, as_tuple=False).squeeze(-1)
        if ready_indices.numel() < 2:
            return None
        if ready_indices.numel() > self.candidate_pool:
            perm = torch.randperm(ready_indices.numel(), device=ready_indices.device)[
                : self.candidate_pool
            ]
            ready_indices = ready_indices[perm]
        feats = self.feature_bank.bank.index_select(0, ready_indices)
        feats = F.normalize(feats, dim=1, eps=1e-12)
        return ready_indices, feats

    def _run_kmeans(self, feats: torch.Tensor):
        k = max(1, min(self.num_clusters, feats.size(0)))
        perm = torch.randperm(feats.size(0), device=self.device)
        centers = feats[perm[:k]].clone()
        assignments = torch.zeros(feats.size(0), dtype=torch.long, device=self.device)
        for _ in range(self.kmeans_iters):
            distances = torch.cdist(feats, centers)
            assignments = torch.argmin(distances, dim=1)
            new_centers = centers.clone()
            for cluster_id in range(k):
                mask = assignments == cluster_id
                if mask.any():
                    new_centers[cluster_id] = feats[mask].mean(dim=0)
                else:
                    rand_idx = torch.randint(0, feats.size(0), (1,), device=self.device)
                    new_centers[cluster_id] = feats[rand_idx]
            centers = F.normalize(new_centers, dim=1, eps=1e-12)
        center_distances = torch.cdist(feats, centers)
        return centers, assignments, center_distances

    def _compute_hardness(self, feats: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        sims = torch.matmul(feats, centers.t()).clamp(-1.0, 1.0)
        topk = min(2, sims.size(1))
        top_vals, _ = torch.topk(sims, k=topk, dim=1)
        if topk == 1:
            raw = 1 - top_vals[:, 0]
        else:
            raw = 1 - top_vals[:, 0] + top_vals[:, 1]
        raw = torch.clamp(raw, min=1e-6)
        return torch.log(raw)

    def _compute_representativeness(self, feats: torch.Tensor) -> torch.Tensor:
        n = feats.size(0)
        if n <= 1:
            return torch.zeros(n, device=self.device)
        neighbors = min(self.neighbor_k + 1, n)
        distances = torch.cdist(feats, feats)
        values, _ = torch.topk(-distances, k=neighbors, dim=1)
        neighbor_dist = (-values)[:, 1:]
        if neighbor_dist.numel() == 0:
            return torch.zeros(n, device=self.device)
        sq_mean = neighbor_dist.pow(2).mean(dim=1) / 2.0
        sq_mean = torch.clamp(sq_mean, min=1e-6)
        return -torch.log(sq_mean)

    def _select_samples(
        self,
        hardness: torch.Tensor,
        representativeness: torch.Tensor,
        feats: torch.Tensor,
        limit: int,
    ) -> List[int]:
        selected: List[int] = []
        available = torch.ones(hardness.size(0), dtype=torch.bool, device=self.device)
        for _ in range(limit):
            if not available.any():
                break
            if selected:
                sel_feats = feats[selected]
                dots = torch.matmul(feats, sel_feats.t()).clamp(-1.0, 1.0)
                diversity = torch.log(torch.clamp(1 - dots, min=1e-6)).min(dim=1).values
            else:
                diversity = torch.zeros_like(hardness)
            scores = hardness + representativeness + diversity
            scores[~available] = float("-inf")
            best_idx = int(torch.argmax(scores).item())
            if not torch.isfinite(scores[best_idx]):
                break
            selected.append(best_idx)
            available[best_idx] = False
        return selected

    def _build_cluster_members(self, assignments, center_distances):
        members = {}
        num_clusters = center_distances.size(1)
        for cluster_id in range(num_clusters):
            mask = assignments == cluster_id
            if not mask.any():
                continue
            positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            dists = center_distances[positions, cluster_id]
            order = torch.argsort(dists)
            members[cluster_id] = positions[order]
        return members

    def _pick_cluster_partner(self, members, cluster_id: int, avoid_pos: int) -> Optional[int]:
        cluster_members = members.get(cluster_id)
        if cluster_members is None or cluster_members.numel() == 0:
            return None
        for pos in cluster_members.tolist():
            if pos != avoid_pos:
                return pos
        return None


def global_pair_feature_loss(
    pairs: List[Tuple[int, int, int]],
    feature_bank: MemoryBank,
    cluster_bank: MemoryBank,
    device: torch.device,
    margin: float,
) -> Optional[torch.Tensor]:
    if not pairs:
        return None
    idx_i = torch.tensor([p[0] for p in pairs], device=device)
    idx_j = torch.tensor([p[1] for p in pairs], device=device)
    labels = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=device)

    feats_i = feature_bank.bank.index_select(0, idx_i)
    feats_j = feature_bank.bank.index_select(0, idx_j)
    feature_loss = F.cosine_embedding_loss(feats_i, feats_j, labels, margin=margin)

    probs_i = cluster_bank.bank.index_select(0, idx_i)
    probs_j = cluster_bank.bank.index_select(0, idx_j)
    pos_mask = labels == 1
    neg_mask = labels == -1
    pos_loss = torch.tensor(0.0, device=device)
    if pos_mask.any():
        pos_loss = F.mse_loss(probs_i[pos_mask], probs_j[pos_mask])
    neg_loss = torch.tensor(0.0, device=device)
    if neg_mask.any():
        divergence = torch.norm(probs_i[neg_mask] - probs_j[neg_mask], dim=1, p=1)
        neg_loss = torch.relu(margin - divergence).mean()
    return feature_loss + 0.5 * (pos_loss + neg_loss)
