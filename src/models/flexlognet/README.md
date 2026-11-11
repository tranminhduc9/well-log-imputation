# FlexLogNet

PyTorch reimplementation of the HGNN/U-Net and FCN heads in Dai, Si and Wu,
[FlexLogNet, Computers & Geosciences 191 (2024), 105666](https://doi.org/10.1016/j.cageo.2024.105666).
Reference code: [Jokercldai/log_completion](https://github.com/Jokercldai/log_completion),
particularly `FlexLogNet_AIgeo10_17/model_AIgeo.py` (time-axis query/key projections).

Each log has independent convolutional encoders and decoders. Inputs include the
observed-normalized signal and median-downsampled, quadratic-interpolated signals
at factors 4, 8 and 16. Four U-Net levels use widths 32, 64, 128 and 256. Decoder
aggregation uses directed non-self edges, per-channel attention and a learned
depth transformation for each ordered log pair. The first aggregation excludes
entirely unobserved source logs; subsequent layers allow reconstructed nodes to
send messages. An FCN predicts mean/std for entirely missing logs.

The small log graph is evaluated with dense PyTorch operations; torch-geometric
and torch-scatter are not required. This also supports variable batch sizes and
the experiment runner's deterministic setting.

## Experiment adaptation

- Windows are 256 points rather than the original 512. GR can be masked, and
  training uses the shared Single/Block-20/Block-100/Entire-Log masks.
- Inputs use the project's train-fitted z-score preprocessing. The paper's
  geological standard-layer preprocessing is not reproduced.
- Within-window normalization uses observed inputs only. Partial logs use their
  observed mean/std for reconstruction; entirely missing logs use FCN estimates.
  Intact target statistics are used only as training labels, never as inputs or
  during prediction.
- The objective combines segment-weighted normalized-gap MSE, masked 1D SSIM
  (Gaussian window 11, normalized data range 2) and statistics MAE for entirely
  hidden logs. The default statistics term is `0.5 * (4 * mean_MAE + std_MAE)`.
  SSIM averages only scored gap positions, rather than zero-padded log images.
- Constant/single-observation logs use unit scale. Predicted scales use softplus
  to stay positive. Inference preserves all observed values.
- Checkpoints are selected by the same mean validation RMSE over four scenarios
  as the other neural models. Results are labeled `FlexLogNet (adapted)`; this is
  not a reproduction of the paper's published scores.
