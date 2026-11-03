"""Train CasFlow with validation early stopping and posterior averaging."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Cascades, collate
from .metrics import report
from .model import CasFlow


@torch.no_grad()
def predict(model, loader, device, seed, samples):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions = []
    for x, lengths, _ in loader:
        prediction, _ = model(x.to(device), lengths, generator=generator, samples=samples)
        predictions.append(prediction.cpu())
    return torch.cat(predictions)


def run(args, prepared, seed):
    output = args.output / f"seed-{seed}"
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    loaders = {
        name: DataLoader(Cascades(tensors), batch_size=args.batch_size, collate_fn=collate,
                         shuffle=name == "train", generator=torch.Generator().manual_seed(seed),
                         pin_memory=device.type == "cuda")
        for name, tensors in prepared["splits"].items()
    }
    val_loader = loaders["val"]
    model = CasFlow().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(seed=seed, torch=str(torch.__version__), numpy=np.__version__,
                  data=prepared["metadata"], parameters=sum(p.numel() for p in model.parameters()))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, best_epoch, stale = float("inf"), 0, 0
    started = time.perf_counter()
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals, count = {}, 0
            for x, lengths, labels in loaders["train"]:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction, terms = model(x, lengths, regularize=True)
                prediction_loss = F.mse_loss(prediction.log1p(), labels.log1p())
                loss = prediction_loss + sum(terms.values())
                loss.backward()
                optimizer.step()
                for name, value in {"loss": loss, "prediction": prediction_loss, **terms}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(labels)
                count += len(labels)
            validation = predict(model, val_loader, device, seed + 10000, args.samples)
            labels = prepared["splits"]["val"]["labels"]
            val_msle = float(F.mse_loss(validation.log1p(), labels.log1p()))
            row = {"epoch": epoch, **{name: value / count for name, value in totals.items()},
                   "val_log1p_msle": val_msle, "seconds": time.perf_counter() - started}
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(f"seed={seed} epoch={epoch} train={row['prediction']:.4f} val={val_msle:.4f}", flush=True)
            if val_msle < best:
                best, best_epoch, stale = val_msle, epoch, 0
                state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                torch.save({"model": state, "config": config, "epoch": epoch}, output / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True, map_location=device)["model"])
    prediction = predict(model, loaders["test"], device, seed + 20000, args.samples)
    test = prepared["splits"]["test"]
    metrics = report(prediction.numpy(), test["labels"].numpy())
    metrics.update(seed=seed, best_epoch=best_epoch, epochs=epoch, val_log1p_msle=best,
                   seconds=time.perf_counter() - started)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "predictions.npz", predictions=prediction.numpy(), labels=test["labels"].numpy(), ids=test["ids"]
    )
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="First training seed.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--samples", type=int, default=8, help="Posterior samples averaged for validation and test."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    prepared = torch.load(args.data, weights_only=True, map_location="cpu")
    results = [run(args, prepared, seed) for seed in range(args.seed, args.seed + args.runs)]
    summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in results])),
            "std": float(np.std([row[metric] for row in results])),
        }
        for metric in ("msle", "mape")
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
