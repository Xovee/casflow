"""Local GraphWave signatures and training-only global graph embeddings."""

import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import torch


def global_embeddings(cascades, dimension=40, seed=0):
    index, rows, columns = {}, [], []
    for cascade in cascades:
        for path, _ in cascade.events:
            for node in path[-2:]:
                if node not in index:
                    index[node] = len(index)
            if len(path) > 1 and path[-1] != path[-2]:
                u, v = index[path[-1]], index[path[-2]]
                rows.extend((u, v))
                columns.extend((v, u))
    n = len(index)
    adjacency = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(n, n))
    del rows, columns
    adjacency.data.fill(1)
    transition = normalize(adjacency, norm="l1", axis=1, copy=False)
    negative = np.asarray(transition.sum(0)).ravel() ** 0.75
    negative /= negative.sum()
    transition.data = np.log(transition.data) - np.log(negative[transition.indices])
    rank = min(dimension, n)
    vectors, singular_values, _ = randomized_svd(transition, n_components=rank, n_iter=5, random_state=seed)
    vectors = normalize(vectors * np.sqrt(singular_values), copy=False)
    vectors[np.diff(adjacency.indptr) == 0] = 0
    if rank < dimension:
        vectors = np.pad(vectors, ((0, 0), (0, dimension - rank)))
    return index, vectors.astype(np.float32, copy=False), {"nodes": n, "edges": adjacency.nnz // 2}


@torch.no_grad()
def local_embeddings(cascades, observation, device, batch_size=32):
    order = 30
    grid = torch.cos(
        (2 * torch.arange(1, order + 1, device=device, dtype=torch.float64) - 1) * torch.pi / (2 * order)
    )
    basis = [torch.ones_like(grid), grid]
    for _ in range(order - 1):
        basis.append(2 * grid * basis[-1] - basis[-2])
    basis = torch.stack(basis)
    time_points = torch.linspace(0, 100, 10, device=device, dtype=torch.float64)
    features = []
    for start in range(0, len(cascades), batch_size):
        batch = [cascade.events[:100] for cascade in cascades[start:start + batch_size]]
        node_lists = [
            list(dict.fromkeys(node for path, _ in events for node in path[-2:])) for events in batch
        ]
        sizes = torch.tensor([len(nodes) for nodes in node_lists], device=device)
        n = int(sizes.max())
        adjacency = np.zeros((len(batch), n, n), dtype=np.float64)
        indices, times = [], []
        for i, (events, nodes) in enumerate(zip(batch, node_lists)):
            index = {node: j for j, node in enumerate(nodes)}
            indices.append([index[path[-1]] for path, _ in events])
            times.append([1 if len(path) == 1 else 1 - arrived / observation for path, arrived in events])
            for path, arrived in events:
                if len(path) > 1 and path[-1] != path[-2]:
                    u, v = index[path[-1]], index[path[-2]]
                    adjacency[i, u, v] = adjacency[i, v, u] = 1 - arrived / observation
        adjacency = torch.as_tensor(adjacency, device=device)
        degree = adjacency.sum(-1)
        inverse = torch.where(degree > 1e-10, degree.rsqrt(), 0)
        operator = -inverse[:, :, None] * adjacency * inverse[:, None, :]
        scales = -torch.log(torch.tensor([0.95, 0.80], device=device, dtype=torch.float64))[None]
        scales = scales * (sizes[:, None].double() / 2).sqrt()
        coefficients = 2 / order * (torch.exp(-scales[:, :, None] * (grid + 1)) @ basis.T)
        coefficients[:, :, 0] /= 2
        previous = torch.eye(n, device=device, dtype=torch.float64).expand(len(batch), n, n)
        current = operator
        heat = coefficients[:, :, 0, None, None] * previous[:, None]
        heat += coefficients[:, :, 1, None, None] * current[:, None]
        for k in range(2, order + 1):
            following = 2 * operator.bmm(current) - previous
            heat += coefficients[:, :, k, None, None] * following[:, None]
            previous, current = current, following
        heat = torch.where(heat > (1e-4 / sizes)[:, None, None, None], heat, 0)
        angles = heat[..., None] * time_points
        valid_rows = (torch.arange(n, device=device)[None] < sizes[:, None])[:, None, :, None, None]
        real = (angles.cos() * valid_rows).sum(2) / sizes[:, None, None, None]
        imaginary = (angles.sin() * valid_rows).sum(2) / sizes[:, None, None, None]
        signatures = torch.stack((real, imaginary), -1).permute(0, 2, 1, 3, 4)
        signatures = signatures.reshape(len(batch), n, 40).float()
        for i, (index, time) in enumerate(zip(indices, times)):
            embedding = signatures[i, index].clone()
            embedding[:, 0] = torch.tensor(time, device=device)
            features.append(embedding.cpu())
        print(f"  embedded {min(start + batch_size, len(cascades))}/{len(cascades)}", flush=True)
    return features
