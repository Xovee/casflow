"""Evaluate a saved CasFlow checkpoint on the prepared test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Cascades, collate
from .metrics import report
from .model import CasFlow
from .train import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    test = torch.load(args.data, weights_only=True, map_location="cpu", mmap=True)["splits"]["test"]
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    model = CasFlow().to(args.device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(Cascades(test), batch_size=config["batch_size"], collate_fn=collate,
                        pin_memory=args.device.startswith("cuda"))
    prediction = predict(model, loader, args.device, config["seed"] + 20000, config["samples"])
    metrics = report(prediction.numpy(), test["labels"].numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "predictions.npz", predictions=prediction.numpy(),
                        labels=test["labels"].numpy(), ids=test["ids"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
