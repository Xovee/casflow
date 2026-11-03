# Implementation differences

The PyTorch version retains CasFlow's local/global structure learning, temporal branch, two-level variational representations, and planar flows. TensorFlow checkpoints and pickles are not supported.

| Part | Archived TensorFlow | PyTorch |
| --- | --- | --- |
| Flows | `K=8` creates 9 layers; Jacobians use updated vectors and cumulative terms are counted repeatedly | 8 layers; evaluate each Jacobian at its input and accumulate once |
| Invertibility | Correction divides by the norm of `w` | Standard correction uses the squared norm |
| Cascade KL | Gaussian KL minus a weighted Jacobian term | Full transformed-posterior KL, including the prior density at the transformed sample |
| Global graph | All raw cascades, including validation/test cascades and later events | Training cascades within their observation window; unseen and isolated nodes receive zero vectors |
| Local graph | Adjacency restricted to adoption-event nodes; self-loops retained | Include implicit parents of the first 100 observed events; remove self-loops from both graphs while retaining events and labels |
| Sequences | Fixed padding affects recurrent computation and losses | Packed sequences; losses over real events, averaged per cascade |
| Normalization | BatchNorm | LayerNorm |
| Node encoder | Stacked linear projections | Nonlinear encoder with 64-dimensional Gaussian latents |
| Predictor | 128/64 hidden units; linear output | 64/32 hidden units; positive Softplus output |
| Evaluation | One stochastic forward pass | Average 8 posterior predictions in `log1p` space using fixed noise |
| Data | Path/time TXT input; pickles, deep copies, Python padding | Lossless Parquet input; contiguous float32 training tensors; pad the current batch only |

## Embeddings and objective

Local GraphWave retains the normalized Laplacian, two scales, order-30 Chebyshev approximation, and 40 features. Global embeddings retain sparse proximity-matrix factorization with 40 features and fix its random seed to 0. Twitter's virtual root `-1` remains a structural node in both graphs; it adds no adoption event or label count.

Planar flows follow [Rezende and Mohamed (2015)](https://proceedings.mlr.press/v37/rezende15.html). For base sample `z0`, transformed sample `zK`, and accumulated `log_det`:

```text
KL(qK || N(0,I)) = KL(q0 || N(0,I))
                  + E[0.5 * (||zK||² - ||z0||²) - log_det]
```

Both levels use standard normal priors. Training adds the two VAEs' reconstruction and KL terms to prediction `log1p` MSE. Terms are averaged over feature/latent dimensions; sequence terms also average real events within each cascade. This retains the public code's two auxiliary VAE objectives; the paper's Eqs. (11)–(13) describe a conditional prior and joint ELBO.

## Data and metrics

The split seed 0, 70/15/15% partition, first 100 input events, Adam learning rate `5e-4`, batch size 64, 1000-epoch cap, and patience 10 are retained. Each event concatenates 40 local and 40 global features; relative arrival time replaces the first local coordinate.

Labels count events with `observation <= elapsed < prediction`, using all events beyond the input limit. Social timestamps use UTC+8. Weibo retains 08:00–18:00 for the half-hour window and 08:00–19:00 for one hour, with exclusive end times. The original comments tie the latter to DeepHawkes/CasCN preprocessing; the paper describes 08:00–18:00. APS retains publication years before 1997.

Training and reporting retain the public code's definitions:

- Train and select checkpoints by natural-log `log1p` squared error.
- Report MSLE after clamping predictions and labels to at least 1, then taking `log2`.
- Report smoothed log-MAPE as `mean(abs(log2(prediction + 1) - log2(label + 1)) / log2(label + 2))`, after the same clamp.

The paper writes an unshifted MAPE formula. The training-only global graph also changes the original experimental protocol, so paper scores serve as reference values.
