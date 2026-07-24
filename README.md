# Dự án Phát hiện Gian lận Tài chính & Hệ thống Scoring Thời gian thực (Fraud Detection System)

Dự án thực hiện quy trình Kỹ thuật Đặc trưng (Feature Engineering), Huấn luyện & So sánh Mô hình Học máy (Random Forest, LightGBM, XGBoost, Logistic Regression), Đánh giá Ngưỡng Chi phí Kinh doanh (Cost Trade-off), cùng Hệ thống **Scoring API (FastAPI)** và **Giao diện Giám sát Real-time Ticker (Streamlit)**.

---

## 📂 Cấu trúc dự án

```text
├── app/
│   ├── api.py                     # FastAPI Scoring Server (Real-time predict, queue, monitoring)
│   ├── features.py                # Feature Extractor tự động 20 đặc trưng từ dữ liệu thô
│   ├── db.py                      # SQLite database lưu trữ queue và lịch sử giao dịch
│   ├── train.py                   # Script huấn luyện & lưu mô hình (Random Forest, LightGBM, XGBoost, LR)
│   ├── evaluation.py              # Đánh giá AUC-PR, Precision/Recall, F1 & Cost-minimizing threshold
│   └── models/                    # Module định nghĩa các mô hình phân loại
├── webapp/
│   ├── app.py                     # Streamlit WebApp (4 Tabs: Results, EDA, Feature List, Realtime Streaming)
│   └── Dockerfile                 # Dockerfile cho Streamlit Frontend
├── Dockerfile.api                 # Dockerfile cho FastAPI Backend Server
├── docker-compose.yml             # Cấu hình containerization phối hợp API & WebApp
├── metadata_features.csv          # Tài liệu mô tả chi tiết các đặc trưng
├── requirements.txt               # Danh sách thư viện phụ thuộc
└── README.md                      # Hướng dẫn sử dụng dự án
```

---

## 🛠️ Hướng dẫn Cài đặt & Khởi động Hệ thống

### 🐳 Cách 1: Chạy bằng Docker Compose (Khuyên dùng - Nhanh nhất)

Chạy lệnh duy nhất tại thư mục gốc của dự án để khởi chạy toàn bộ Backend API và Streamlit WebApp:

```bash
docker-compose up --build
```

Sau khi khởi chạy thành công:
- **Streamlit WebApp Interface**: Truy cập `http://localhost:8501`
- **FastAPI Interactive API Docs**: Truy cập `http://localhost:8000/docs`

---

### 🐍 Cách 2: Chạy trực tiếp bằng Python (Virtual Environment)

#### Bước 1: Kích hoạt Môi trường ảo & Cài đặt Thư viện
```bash
# Kích hoạt venv (trên macOS/Linux)
source venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

#### Bước 2: Huấn luyện & Lưu Mô hình (Nếu chưa có `model/model.joblib`)
```bash
PYTHONPATH=. python app/train.py --model rf --params '{"n_estimators": 50, "max_depth": 10}' --model-out model/model.joblib
```

#### Bước 3: Khởi động Backend API & Streamlit Dashboard

- **Terminal 1: Mở FastAPI Backend Server**
  ```bash
  PYTHONPATH=. python -m uvicorn app.api:app --reload --port 8000
  ```

- **Terminal 2: Mở Streamlit Frontend Dashboard**
  ```bash
  streamlit run webapp/app.py --server.port 8501
  ```

---

### ⚙️ Cách 3: Cài đặt & Chạy riêng lẻ Backend API (FastAPI Standalone)

Nếu bạn chỉ muốn chạy riêng dịch vụ Backend API để tích hợp hoặc kiểm thử các endpoint API độc lập:

#### Phương án 3.1: Chạy riêng Backend qua Docker
```bash
# Cách A: Chạy bằng Docker Compose chỉ dịch vụ Backend API
docker-compose up --build api

# Cách B: Build & Run trực tiếp bằng Dockerfile.api
docker build -f Dockerfile.api -t fraud-detection-api .
docker run -d -p 8000:8000 -v $(pwd)/model:/app/model --name fraud-api fraud-detection-api
```

#### Phương án 3.2: Chạy riêng Backend qua Môi trường ảo Python
```bash
# 1. Kích hoạt môi trường ảo
source venv/bin/activate  # Hoặc source .venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 3. Khởi động riêng FastAPI Server
PYTHONPATH=. python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

#### 📌 Truy cập các Endpoints của Backend API:
- **Swagger Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc API Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`
- **Scoring Endpoint (`POST`)**: `http://localhost:8000/predict`
- **Batch Push Endpoint (`POST`)**: `http://localhost:8000/push-data`
- **Review Queue Endpoint (`GET`)**: `http://localhost:8000/queue`
- **Data Drift & Retraining Triggers**: `http://localhost:8000/monitoring/drift`

---

## 🖥️ Các Tính năng Chính trên Giao diện Streamlit Dashboard

1. **📊 Tab 1 - Model Results**: Báo cáo chỉ số đánh giá mô hình (Precision 99.37%, Recall 99.98%, AUC-PR 0.9998), Ma trận Nhầm lẫn (Confusion Matrix) và Bảng so sánh hiệu năng 4 mô hình (**Random Forest**, **LightGBM**, **XGBoost**, **Logistic Regression**).
2. **📁 Tab 2 - Data Source & EDA**: Xem thông tin tổng quan bộ dữ liệu, bản xem trước dữ liệu thô (Raw Data Preview) và các phát hiện dị thường quan trọng từ EDA.
3. **🧬 Tab 3 - Feature List**: Bảng chi tiết 20 đặc trưng (`SELECTED_FEATURES`), công thức tính toán và ý nghĩa nghiệp vụ nhận diện dị thường.
4. **⚡ Tab 4 - Real-time Streaming**:
   - **Upload File CSV & Stream 3 bản ghi/giây**: Upload file CSV giao dịch bất kỳ, ứng dụng sẽ đẩy tuần tự 3 bản ghi/giây vào 4 mô hình kèm thanh tiến trình trực quan.
   - **Tải file CSV mẫu nhanh (`📥 Download Sample Streaming CSV`)**: Cung cấp nút tải nhanh file dữ liệu giao dịch mẫu để trải nghiệm streaming ngay lập tức.
   - **Nút đẩy dữ liệu ngẫu nhiên thời gian thực**: (`Push 1 Random Transaction`, `Push 10 Random Transactions`).
   - **Công tắc Giả lập Continuous Real-time (`▶️ Enable Continuous Auto-Stream`)**: Tự động sinh ngẫu nhiên 1-3 giao dịch sau mỗi 1-5 giây.
   - **Biểu đồ Ticker Multi-Color kiểu Chứng khoán**: Vẽ 4 đường đồ thị biến động xác suất rủi ro (`Fraud Score %`) của cả 4 mô hình song song thời gian thực.

