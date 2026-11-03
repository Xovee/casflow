# CasFlow

PyTorch reference implementation of [CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction](https://doi.org/10.1109/TKDE.2021.3126475), TKDE 2021.

## Run

Use Python 3.12 and a PyTorch build suitable for your hardware.

```bash
pip install -r requirements.txt

python -m casflow.prepare --dataset weibo \
    --raw datasets/cascades/weibo.parquet --output prepared/weibo.pt
python -m casflow.train --data prepared/weibo.pt --output results/weibo
```

Download [cascades.zip](https://drive.google.com/file/d/1r-OHiiO9QdYH8sBc6tjmOd-vb_rJCYBo/view?usp=sharing) and extract it into `datasets/`. A small Weibo example is included at `datasets/sample.parquet`; see the [data format](datasets/README.md).

Set `--observation` during preparation; the default is the first window below. Weibo and Twitter use seconds; APS uses days.

| Dataset | Observation windows | Prediction horizon |
| --- | --- | --- |
| Weibo | 1800 / 3600 | 86400 |
| Twitter | 86400 / 172800 | 2764800 |
| APS | 1095 / 1826 | 7305 |

The train/validation/test split is fixed at 70/15/15%. Global embeddings use only training cascades within their observation window. Validation/test cascades do not contribute to this graph; unseen or isolated nodes receive zero global vectors.

Training stops after 10 epochs without validation improvement and saves the best model, history, test predictions, and metrics. Use `--runs N` for independent repetitions, starting at `--seed` (default: 0). `summary.json` reports means and standard deviations. CPU/CUDA selection is automatic, or set with `--device`.

## Test

Training evaluates the test split automatically. To evaluate a saved checkpoint:

```bash
python -m casflow.test --data prepared/weibo.pt \
    --checkpoint results/weibo/seed-0/best.pt --output results/weibo/seed-0/test
```

Evaluation reuses the checkpoint's seed, batch size and posterior sample count, and saves metrics and predictions.

The original TensorFlow code is archived in `legacy/tensorflow/`. See [implementation differences](IMPLEMENTATION-DIFFERENCES.md).

## Citation

```bibtex
@article{xu2023casflow,
  author = {Xovee Xu and Fan Zhou and Kunpeng Zhang and Siyuan Liu and Goce Trajcevski},
  title = {Cas{F}low: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year = {2023},
  volume = {35},
  number = {4},
  pages = {3484--3499},
  doi = {10.1109/TKDE.2021.3126475}
}
```

Released under the [MIT License](LICENSE).
