# Dự án Kỹ thuật Đặc trưng Phát hiện Gian lận Tài chính (Fraud Detection Feature Engineering)

Dự án này thực hiện quy trình tiền xử lý dữ liệu, kiểm tra chất lượng và xây dựng hệ thống đặc trưng (feature engineering) từ tập dữ liệu giao dịch tài chính mô phỏng (Synthetic Financial Datasets). Mục tiêu là tạo ra tập dữ liệu chất lượng cao để huấn luyện các mô hình học máy phát hiện giao dịch gian lận.

---

## 📂 Cấu trúc thư mục dự án

```text
├── 1_clean_data.ipynb             # Notebook làm sạch dữ liệu & phân tích ngoại lệ
├── 2_feature_engineer.ipynb       # Notebook xây dựng đặc trưng (Feature Engineering)
├── metadata_features.csv          # Tài liệu mô tả chi tiết ý nghĩa của 48 đặc trưng
├── requirements.txt               # Danh sách các thư viện cần thiết
├── Synthetic_Financial_datasets_log.csv       # Tập dữ liệu giao dịch gốc (CSV)
└── Synthetic_Financial_datasets_features.parquet # Tập dữ liệu sau khi biến đổi đặc trưng (Parquet)
```

---

## 🛠️ Hướng dẫn cài đặt môi trường

Để chạy dự án này một cách ổn định và tránh xung đột thư viện, bạn nên sử dụng môi trường ảo (virtual environment).

### Bước 1: Kích hoạt môi trường ảo (Virtual Environment)
Nếu bạn đã có thư mục `venv` trong dự án:

- **Trên macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Trên Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate
  ```
- **Trên Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

*(Nếu chưa có, bạn có thể tạo mới bằng lệnh: `python3 -m venv venv`)*

### Bước 2: Cài đặt các thư viện phụ thuộc
Sau khi kích hoạt môi trường ảo, chạy lệnh sau để cài đặt các thư viện từ `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Đăng ký môi trường ảo với Jupyter Notebook
Đăng ký kernel của môi trường ảo để Jupyter Notebook có thể nhận diện chính xác các thư viện đã cài đặt:
```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

---

## 🚀 Hướng dẫn chạy dự án

### Bước 1: Khởi động Jupyter Notebook
Chạy lệnh dưới đây tại thư mục gốc của dự án:
```bash
jupyter notebook
```
Hoặc nếu sử dụng JupyterLab:
```bash
jupyter lab
```

### Bước 2: Chạy các Notebook theo thứ tự

#### 1. `1_clean_data.ipynb`
* **Nhiệm vụ:**
  - Kiểm tra kiểu dữ liệu, các giá trị bị thiếu (missing values) và trùng lặp (duplicates).
  - Phân tích các giá trị ngoại lệ (outliers) bằng phương pháp IQR.
  - Kiểm tra tính nhất quán và tính hợp lệ của logic nghiệp vụ số dư tài khoản.
  - Loại bỏ các cột không cần thiết (ví dụ: `isFlaggedFraud`).
* **Cách thực hiện:** Mở file và chọn **Kernel** -> **Restart & Run All Cells**. Chọn kernel là `"Python (venv)"` mà bạn đã đăng ký ở trên.

#### 2. `2_feature_engineer.ipynb`
* **Nhiệm vụ:**
  - Sắp xếp dữ liệu theo trình tự thời gian (`step`) để tránh rò rỉ thông tin từ tương lai.
  - Mã hóa One-Hot (One-Hot Encoding) cho các loại hình giao dịch và gắn nhãn tài khoản Merchant (`is_merchant_dest`).
  - Xây dựng các đặc trưng thời gian có tính chu kỳ (`hour_of_day`, `day_of_month`, `day_of_week`).
  - Xây dựng các đặc trưng chênh lệch số dư (`errorBalanceOrig`, `errorBalanceDest`), tỷ lệ số tiền rút/chuyển, các chỉ báo số dư bằng 0.
  - Xây dựng đặc trưng lịch sử lũy kế (cumulative history) và vận tốc giao dịch (sliding window velocity) trong 24 giờ qua của tài khoản gửi và tài khoản nhận.
  - Phân tích mối tương quan của các đặc trưng mới tạo với nhãn mục tiêu `isFraud`.
  - Lưu kết quả cuối cùng dưới định dạng nén Parquet (`Synthetic_Financial_datasets_features.parquet`) để tối ưu hóa bộ nhớ và tốc độ đọc/ghi.
* **⚠️ Lưu ý quan trọng:**
  - Phép tính trượt 24h và tính lũy kế trên tập dữ liệu đầy đủ (~6.3 triệu dòng) là những phép toán tính toán nặng.
  - Thời gian thực thi toàn bộ notebook `2_feature_engineer.ipynb` có thể mất khoảng **2 giờ (~7.300 giây)** tùy thuộc vào sức mạnh xử lý của CPU. Vui lòng đảm bảo máy tính không bị tắt nguồn hay rơi vào trạng thái ngủ (sleep) trong lúc chạy.

---

## 📊 Kết quả đầu ra và Tài liệu Đặc trưng

- **Tập dữ liệu đầu ra:** `Synthetic_Financial_datasets_features.parquet` (~800 MB). Đây là định dạng lưu trữ cột tối ưu, chứa tất cả các đặc trưng gốc và đặc trưng mới được kỹ nghệ hóa.
- **Tài liệu đặc trưng:** Bạn có thể tham khảo file [metadata_features.csv](file:///Users/hltlong/Project/master/BA_project/metadata_features.csv) để xem danh sách chi tiết cùng định nghĩa, ý nghĩa nghiệp vụ và tầm quan trọng đối với mô hình của từng đặc trưng trong tổng số 48 đặc trưng.

## Model training and experiment tracking

The command-line training pipeline supports Random Forest, XGBoost, and LightGBM with local MLflow tracking or optional Weights & Biases tracking. See [the W&B training guide](docs/6_wandb_training.md) for setup, online/offline modes, and commands for all models.
