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

## 🖥️ Các Tính năng Chính trên Giao diện Streamlit Dashboard

1. **📊 Tab 1 - Model Results**: Báo cáo chỉ số đánh giá mô hình (Precision 99.37%, Recall 99.98%, AUC-PR 0.9998), Ma trận Nhầm lẫn (Confusion Matrix) và Bảng so sánh hiệu năng 4 mô hình (**Random Forest**, **LightGBM**, **XGBoost**, **Logistic Regression**).
2. **📁 Tab 2 - Data Source & EDA**: Xem thông tin tổng quan bộ dữ liệu, bản xem trước dữ liệu thô (Raw Data Preview) và các phát hiện dị thường quan trọng từ EDA.
3. **🧬 Tab 3 - Feature List**: Bảng chi tiết 20 đặc trưng (`SELECTED_FEATURES`), công thức tính toán và ý nghĩa nghiệp vụ nhận diện dị thường.
4. **⚡ Tab 4 - Real-time Streaming**:
   - Nút đẩy dữ liệu ngẫu nhiên thời gian thực (`Push 1 Random Transaction`, `Push 10 Random Transactions`).
   - **Công tắc Giả lập Continuous Real-time (`▶️ Enable Continuous Auto-Stream`)**: Tự động sinh ngẫu nhiên 1-3 giao dịch sau mỗi 1-5 giây.
   - **Biểu đồ Ticker Multi-Color kiểu Chứng khoán**: Vẽ 4 đường đồ thị biến động xác suất rủi ro (`Fraud Score %`) của cả 4 mô hình song song thời gian thực.
