# Hướng dẫn Train, Cấu hình Tham số, và Theo dõi Kết quả (MLflow)

## 1. Chạy train cơ bản (auto hyperparameter search)

```bash
uv run python -m app.train --model rf --n-iter 25
```

- `--model`: `rf` | `xgboost` | `lightgbm` (đăng ký trong `MODEL_REGISTRY`, `app/train.py`).
- `--n-iter`: số lần thử của `RandomizedSearchCV` (`app/tuning.py::time_series_search`), mặc định 25.
- Search dùng `TimeSeriesSplit(n_splits=5)`, `scoring="average_precision"` — không random-shuffle vì dữ liệu có tính thời gian (tránh leakage).
- Không gian tham số search nằm ở `param_distributions()` trong từng file model (`app/models/rf.py`, `xgboost_model.py`, `lightgbm_model.py`) — sửa ở đó nếu muốn mở rộng/thu hẹp không gian tìm kiếm.

## 2. Train với tham số cố định (bỏ qua search)

Dùng khi muốn tự set và giữ nguyên một bộ tham số cụ thể để so sánh kết quả trực tiếp, không phụ thuộc vào việc search ngẫu nhiên chọn gì:

```bash
uv run python -m app.train --model rf --params '{"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 2, "max_features": "sqrt"}'
```

- Cờ `--params` nhận một chuỗi JSON, `train.py` sẽ `set_params(**parsed)` lên estimator rồi fit trực tiếp trên `X_train_selected, y_train` — **không** chạy `RandomizedSearchCV`.
- Khi dùng `--params`, `best_cv_score` sẽ là `None` (không có CV score vì không search) — vẫn có đầy đủ metrics trên test set, cost curve, threshold tối ưu.
- Tên tham số phải khớp đúng tên tham số của estimator tương ứng (vd RF: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`; XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`; LightGBM: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`).

## 3. Các cờ khác

| Cờ | Ý nghĩa |
|---|---|
| `--plot-learning-curve` | Vẽ learning curve (`sklearn.model_selection.learning_curve`, `cv=TimeSeriesSplit(n_splits=5)`) và log làm artifact vào MLflow. Tốn thêm thời gian đáng kể (refit nhiều lần theo từng train-size fraction). |
| `--model-out PATH` | Lưu thêm một bản model độc lập bằng `joblib.dump` ra `PATH`, ngoài bản MLflow đã tự log sẵn (không bắt buộc, chỉ cần khi muốn có file model tách rời ngoài `mlruns/`). |

Ví dụ đầy đủ:

```bash
uv run python -m app.train \
  --model xgboost \
  --params '{"n_estimators": 60, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}' \
  --plot-learning-curve \
  --model-out models/xgboost_manual_v1.joblib
```

## 4. Chạy nền (background) cho các lần train lâu

Trên tập đầy đủ (~5M dòng train), RF/XGBoost với nhiều `n_estimators` hoặc `--plot-learning-curve` có thể chạy vài phút đến hơn chục phút. Khuyến nghị chạy nền:

```bash
nohup uv run python -m app.train --model rf --n-iter 5 --plot-learning-curve > /tmp/rf_run.log 2>&1 &
echo "PID: $!"
```

Theo dõi log: `tail -f /tmp/rf_run.log`. Dừng sớm nếu cần: `kill <PID>`.

## 5. Theo dõi kết quả trong MLflow

Mỗi lần chạy `app/train.py` tự động gọi `mlflow_utils.init_tracking()` + `mlflow_utils.log_run()`:

- Tracking URI: `file:<project_root>/mlruns` (local file store, đã set `MLFLOW_ALLOW_FILE_STORE=true`).
- Log params: toàn bộ `best_params` (từ search hoặc từ `--params`).
- Log metrics: tất cả metric numeric trong `metrics_best_threshold` (đã tính tại ngưỡng tối ưu chi phí) — `precision`, `recall`, `f1`, `auc_pr`, `roc_auc`, các ô `confusion_matrix_tn/fp/fn/tp`, và `best_threshold`.
- Log model: `mlflow.sklearn.log_model(model, artifact_path="model")` — có thể load lại bằng `mlflow.sklearn.load_model(...)`.
- Nếu có `--plot-learning-curve`: file PNG learning curve được log làm artifact riêng.

### Xem bằng MLflow UI

```bash
uv run mlflow ui --backend-store-uri file:./mlruns
```

Mở `http://localhost:5000`, chọn experiment mặc định (`experiment_id=0`) để xem danh sách run, so sánh metric giữa các run (vd so `auc_pr` giữa run search tự động và run `--params` thủ công).

### Xem nhanh bằng script (không cần mở UI)

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
runs = mlflow.search_runs(experiment_ids=['0'])
print(runs[['tags.mlflow.runName', 'params.n_estimators', 'metrics.auc_pr', 'metrics.best_threshold']].tail(10))
"
```

## 6. Tóm tắt: quy trình đề xuất khi muốn thử một cấu hình cụ thể

1. Chọn model (`--model`) và bộ tham số muốn thử.
2. Chạy với `--params '{...}'` (bỏ qua search) để kết quả tái lập được (deterministic, không phụ thuộc random search).
3. Thêm `--plot-learning-curve` nếu muốn kiểm tra overfit/underfit theo train-size.
4. Chạy nền nếu ước tính lâu (`nohup ... &`), theo dõi log.
5. Sau khi chạy xong, mở MLflow UI hoặc script trên để so sánh `auc_pr`, `best_threshold`, confusion matrix giữa các lần chạy — chọn cấu hình tốt nhất dựa trên số liệu thực tế, không chỉ dựa vào CV score (vì `--params` không có CV score).
