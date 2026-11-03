"""Prepare train/validation/test tensors from Parquet cascades."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .data import SETTINGS, read_cascades, split_cascades
from .graph import global_embeddings, local_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SETTINGS)
    parser.add_argument("--raw", required=True, type=Path, help="A cascade Parquet file.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observation", type=int, help="Seconds for social data; days for APS.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source = json.loads(pq.ParquetFile(args.raw).schema_arrow.metadata[b"cascade_metadata"])
    if source["dataset"] != args.dataset:
        parser.error("--dataset must match the dataset recorded in the Parquet file.")
    observation = args.observation or SETTINGS[args.dataset][0][0]
    cascades, digest = read_cascades(args.raw, args.dataset, observation)
    splits = split_cascades(cascades)
    print(f"Cascades: { {name: len(items) for name, items in splits.items()} }", flush=True)
    index, vectors, graph_info = global_embeddings(splits["train"])
    prepared = {}
    for name, items in splits.items():
        print(f"Embedding {name}", flush=True)
        local = local_embeddings(items, observation, args.device)
        global_features, known = [], 0
        for cascade in items:
            features = np.zeros((min(100, len(cascade.events)), 40), dtype=np.float32)
            for i, (path, _) in enumerate(cascade.events[:100]):
                if path[-1] in index:
                    features[i] = vectors[index[path[-1]]]
                    known += 1
            global_features.append(torch.from_numpy(features))
        lengths = torch.tensor([len(x) for x in local])
        prepared[name] = {
            "features": torch.cat([torch.cat((cg, gg), -1) for cg, gg in zip(local, global_features)]),
            "lengths": lengths,
            "labels": torch.tensor([c.label for c in items], dtype=torch.float32),
            "ids": [c.id for c in items],
        }
        graph_info[f"{name}_node_coverage"] = known / int(lengths.sum())
    metadata = {
        "dataset": args.dataset, "observation": observation, "prediction": SETTINGS[args.dataset][1],
        "split_seed": 0, "raw_sha256": digest, "source": source, "global_graph": graph_info,
        "weibo_end_hour": (19 if observation == 3600 else 18) if args.dataset == "weibo" else None,
        "torch": str(torch.__version__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "splits": prepared}, args.output)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
