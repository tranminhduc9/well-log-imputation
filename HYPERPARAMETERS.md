# Tổng hợp hyperparameters

Tài liệu này tổng hợp các giá trị đang được sử dụng bởi `main.py`, `cfg.py` và
10 model xuất hiện trong `notebook.ipynb`. Các bảng bên dưới chỉ ghi những tham
số được project thiết lập rõ ràng; tham số không được truyền vào thư viện sẽ
dùng mặc định của phiên bản thư viện được cài đặt.

## 1. Model khả dụng

| Tên CLI | Model | File cấu hình/implementation | Alias |
|---|---|---|---|
| `locf` | Last Observation Carried Forward | `models/locf.py` | — |
| `rf` | Random Forest | `models/random_forest.py` | — |
| `qrf` | Quantile Random Forest | `models/quantile_random_forest.py` | `quantilerf` |
| `bayesnn` | Bayesian Neural Network | `models/bayesian_nn.py` | `bayessnn` |
| `xgboost` | XGBoost | `models/xgboost.py` | — |
| `unet` | 1D U-Net | `models/unet.py` | — |
| `saits` | SAITS | `models/saits.py` | — |
| `np` | Neural Process (mean aggregation) | `models/neural_process.py` | — |
| `anp_standard` | Attentive Neural Process không depth penalty | `models/attentive_neural_process.py` | — |
| `anp` | Depth-aware Attentive Neural Process | `models/attention_neural_process.py` | — |

## 2. Cấu hình mặc định khi chạy `main.py`

Đây là các giá trị mặc định thực tế từ `cfg.py`.

### 2.1 Runtime và dữ liệu

| Tham số CLI | Mặc định | Phạm vi/ý nghĩa |
|---|---:|---|
| `--output_dir` | `trained_models` | Thư mục log, trọng số và prediction artifacts |
| `--n_folds` | `5` | Số fold cross-validation |
| `--seed`, `-s` | `17076` | Seed dùng cho bộ sinh số ngẫu nhiên |
| `--dataset_name` | `geolink` | Một trong `geolink`, `taranaki`, `teapot` |
| `--dataset_dir` | `<project>/imputation-processed-datasets` | Thư mục chứa các file `.npy` đã xử lý |
| `--logs` | `GR DTC RHOB NPHI` | Danh sách feature theo đúng thứ tự trong array |
| `n_features` | `4` | Được suy ra bằng `len(logs)`, không truyền trực tiếp qua CLI |
| `--model` | `saits` | Model được chọn; xem danh sách ở mục 1 |
| `--device` | `gpu` | `gpu` hoặc `cpu`; tự chuyển thành `cuda` nếu CUDA khả dụng, ngược lại dùng `cpu` |

### 2.2 Huấn luyện dùng chung

| Tham số CLI | Mặc định | Model sử dụng |
|---|---:|---|
| `--slice_len` | `256` | SAITS, U-Net, họ NP; được truyền thành `seq_len`/`n_steps` |
| `--epochs` | `500` | SAITS, U-Net, BayesNN, họ NP |
| `--patience` | `50` | Early stopping của SAITS, U-Net, BayesNN, họ NP |
| `--batch_size` | `32` | SAITS, U-Net, họ NP; BayesNN áp dụng quy tắc riêng bên dưới |
| `--lr` | `1e-3` | SAITS/U-Net qua optimizer; BayesNN dùng trực tiếp; họ NP giới hạn tối đa `3e-4` |
| `--optimizer` | `adam` | `adam` hoặc `adamw`; chỉ SAITS và U-Net lấy optimizer từ CLI |

### 2.3 Missing-pattern experiments

| Tham số CLI | Mặc định | Ý nghĩa |
|---|---:|---|
| `--missing_pattern` | `single block profile` | Các kiểu missing được đánh giá |
| `--n_points` | `1` | Số điểm bị mask trong mỗi sequence cho pattern `single` |
| `--blocks_size` | `20 100` | Các độ dài block liên tiếp bị mask |
| `--profiles` | `RAND` | Random một feature cho mỗi sequence; cũng có thể chỉ định tên log |

Trong training, mode `rand` chọn ngẫu nhiên một trong `single`, `block` hoặc
`profile` cho từng sequence. Với `block`, kích thước được chọn từ `20` hoặc
`100`; với `single`, số điểm mặc định là `1`.

## 3. Hyperparameters theo model

### 3.1 LOCF

| Tham số | Giá trị |
|---|---:|
| `first_step_imputation` | `zero` |
| Có training | Không |

Các missing value đứng đầu sequence, nơi không có quan sát trước đó để carry
forward, được điền bằng `0`.

### 3.2 Random Forest (`rf`)

Project huấn luyện một `RandomForestRegressor` độc lập cho mỗi feature.

| Tham số | Giá trị |
|---|---:|
| `num_models` | `n_features` |
| `n_estimators` | `200` |
| `max_depth` | `None` |
| `min_samples_leaf` | `1` |
| `min_samples_split` | `2` |
| `random_state` | Không truyền; dùng mặc định của scikit-learn |

### 3.3 Quantile Random Forest (`qrf`, `quantilerf`)

Project huấn luyện một forest cho mỗi feature. Point prediction là median của
phân phối prediction từ các tree.

| Tham số | Giá trị |
|---|---:|
| `num_models` | `n_features` |
| `n_estimators` | `100` |
| `max_depth` | `20` |
| `min_samples_leaf` | `5` |
| `min_samples_split` | `10` |
| `max_samples` | `0.5` |
| `n_jobs` | `-1` |
| `random_state` | `17076` |
| `lower_quantile` | `0.05` |
| Point quantile | `0.50` |
| `upper_quantile` | `0.95` |
| `prediction_chunk_size` | `50,000` |
| Prediction interval | `90%` |

### 3.4 XGBoost (`xgboost`)

Project huấn luyện một `XGBRegressor` độc lập cho mỗi feature.

| Tham số | Giá trị |
|---|---:|
| `num_models` | `n_features` |
| `n_estimators` | `26` |
| `min_child_weight` | `6.0` |
| `gamma` | `0.0` |
| `subsample` | `0.8` |
| `colsample_bytree` | `1.0` |
| `reg_alpha` | `0.0` |
| `learning_rate` | `0.1` |
| `random_state` | Không truyền; dùng mặc định của XGBoost |

Lưu ý: `learning_rate=0.1` của XGBoost là giá trị cố định trong model và không
dùng `--lr` của CLI.

### 3.5 SAITS (`saits`)

| Nhóm | Tham số | Giá trị mặc định hiệu lực |
|---|---|---:|
| Input | `n_steps` | `slice_len = 256` |
| Input | `n_features` | `len(logs) = 4` |
| Kiến trúc | `n_layers` | `2` |
| Kiến trúc | `d_model` | `256` |
| Kiến trúc | `d_inner` | `128` |
| Attention | `n_heads` | `4` |
| Attention | `d_k` | `64` |
| Attention | `d_v` | `64` |
| Regularization | `dropout` | `0.1` |
| Regularization | `attn_dropout` | `0.1` |
| Attention | `diagonal_attention_mask` | `True` |
| Loss | `ORT_weight` | `1` |
| Loss | `MIT_weight` | `1` |
| Training | `batch_size` | `32` |
| Training | `epochs` | `500` |
| Training | `patience` | `50` |
| Training | `optimizer` | `Adam(lr=1e-3)` |
| Runtime | `num_workers` | `0` |
| Checkpoint | `model_saving_strategy` | `best` |

Nếu CLI chọn `--optimizer adamw`, optimizer hiệu lực trở thành
`AdamW(lr=1e-3)`.

### 3.6 1D U-Net (`unet`)

| Nhóm | Tham số | Giá trị mặc định hiệu lực |
|---|---|---:|
| Input/output | `in_channels` | `n_features = 4` |
| Input/output | `out_channels` | `n_features = 4` |
| Kiến trúc | `spatial_dims` | `1` |
| Kiến trúc | `channels` | `(32, 64, 128, 256, 512)` |
| Kiến trúc | `strides` | `(2, 2, 2, 2)` |
| Kiến trúc | `num_res_units` | `1` |
| Training | `batch_size` | `32` |
| Training | `epochs` | `500` |
| Training | `patience` | `50` |
| Training | `optimizer` | `Adam(lr=1e-3)` |
| Runtime | `num_workers` | `0` |
| Checkpoint | `model_saving_strategy` | `best` |
| Loss | Reconstruction loss | Masked MSE |
| Loss | Observed/reconstruction weight | `1` |
| Loss | Artificial-missing imputation weight | `5` |

Tổng loss của U-Net là `L_rec + 5 × L_imp`. Nếu CLI chọn
`--optimizer adamw`, optimizer hiệu lực trở thành `AdamW(lr=1e-3)`.

### 3.7 Bayesian Neural Network (`bayesnn`, `bayessnn`)

Project huấn luyện một variational BNN cho mỗi feature, theo Feng et al. (2021).
Mỗi model dùng các feature còn lại làm input; mọi weight và bias đều có posterior
Gaussian học được bằng reparameterization, thay vì MC Dropout.

| Nhóm | Tham số | Giá trị mặc định hiệu lực |
|---|---|---:|
| Số model | `num_models` | `n_features = 4` |
| Kiến trúc | Hidden layers | `2` |
| Kiến trúc | `hidden_size` mỗi layer | `10` |
| Kiến trúc | Activation | `ReLU` |
| Kiến trúc | Output size | `1` |
| Variational posterior | Family | Factorized Gaussian `N(mu, softplus(rho)^2)` |
| Khởi tạo posterior | `rho` | `0` |
| Prior Type I | `pi`, `sigma1`, `sigma2` | `0.5`, `1.5`, `0.1` |
| Loss | Variational free energy | `KL(q || p) / n_train + Gaussian NLL` |
| Training | `batch_size` | `max(CLI batch size, 4096) = 4096` |
| Validation | Validation batch size | `max(model batch size, 1024) = 4096` |
| Training | `epochs` | `CLI epochs = 500` |
| Training | `patience` | `CLI patience = 50` |
| Training | `learning_rate` | `--lr = 1e-3` |
| Training | Optimizer | PyTorch `Adam` |
| Early stopping | `min_delta` | `1e-4` |
| Inference | `mc_samples` | `1000` |
| Inference | `prediction_batch_size` | `8192` |
| Prediction | Point estimate | Ensemble mean |
| Uncertainty | `std` | Empirical ensemble standard deviation |
| Interval | `lower`, `upper` | Ensemble mean `+- 1 std` |

BayesNN dùng `--lr` trực tiếp nhưng không dùng lựa chọn `--optimizer`; optimizer
luôn là PyTorch `Adam`.

### 3.8 Neural Process family (`np`, `anp_standard`, `anp`)

| Nhóm | Tham số | Giá trị mặc định hiệu lực |
|---|---|---:|
| Input | `n_steps` | `slice_len = 256` |
| Input | `n_features` | `len(logs) = 4` |
| Kiến trúc | `hidden_dim` | `128` |
| Kiến trúc | `latent_dim` | `32` |
| Attention (`anp_standard`, `anp`) | `n_heads` | `4` |
| Depth penalty (chỉ `anp`) | `initial_depth_scale` | `0.2` |
| Regularization | `dropout` | `0.1` |
| Training | `batch_size` | `32` |
| Training | `epochs` | `500` |
| Training | `patience` | `50` |
| Training | `learning_rate` | `min(--lr, 3e-4) = 3e-4` |
| Training | Optimizer | PyTorch `AdamW` |
| Training | `weight_decay` | `1e-5` |
| Loss | `kl_weight` | `3e-3`, warm-up trong 40 epoch |
| Loss | `observed_loss_weight` | `0.1` |
| Numerical bound | `scale` | Clamp trong `[0.03, 3.0]` |
| Training mask | Missing pattern | Dùng dataset `rand` chung do `main.py` tạo |
| Checkpoint | Selection metric | MAE trên các vị trí bị mask |
| Runtime | `num_workers` | `0` |
| Inference | Latent samples | `16` mẫu từ prior |
| Prediction interval | z-score | `1.6448536269514722` |
| Prediction interval | Coverage | `90%` |

Ba model giữ nguyên latent encoder, decoder, Gaussian head, loss, lịch KL,
optimizer và inference sampling. Khác biệt duy nhất nằm ở deterministic path:

- `np`: mean pooling các context representation, không có attention;
- `anp_standard`: scaled dot-product multi-head cross-attention, không cộng
  penalty theo khoảng cách depth;
- `anp`: cùng cross-attention nhưng score trừ thêm
  `|depth_query - depth_key| / learned_depth_scale` cho từng head.

Các MLP dùng hai hidden layer kích thước `hidden_dim`, activation `ReLU`, và
dropout `0.1`. Họ NP giới hạn learning rate ở `3e-4` để ổn định Gaussian
variance head; optimizer luôn là PyTorch `AdamW`, không phụ thuộc lựa chọn
`--optimizer`.

## 4. Mặc định khi gọi `ModelFactory` trực tiếp

Khi chạy qua CLI, các giá trị ở mục 2 được truyền vào factory. Nếu gọi
`ModelFactory(...)` trực tiếp mà không truyền các tham số tương ứng, factory dùng:

| Tham số | Mặc định của `ModelFactory` |
|---|---:|
| `seq_len` | `256` |
| `n_features` | `4` |
| `batch_size` | `32` |
| `epochs` | `50` |
| `patience` | `50` |
| `optimizer` | `None` |
| `learning_rate` | `1e-3` |
| `device` | `cpu` |
| `output_dir` | `.` |

Do đó, để tái lập đúng cấu hình notebook/CLI, nên chạy qua `main.py` hoặc truyền
đầy đủ `epochs=500`, `patience=50` và optimizer tương ứng khi dùng factory trực
tiếp.
