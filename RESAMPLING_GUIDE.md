# Tài liệu Tích hợp: Xử lý Mất cân bằng Dữ liệu (Hybrid Resampling)

## 1. Tổng quan Module
Module resampling.py cung cấp giải pháp xử lý mất cân bằng cực đại (extreme imbalance) cho tập dữ liệu 6.3 triệu dòng. Việc sử dụng SMOTE trực tiếp sẽ gây tràn bộ nhớ (OOM). Do đó, hệ thống áp dụng luồng Hybrid Pipeline của imblearn:

RandomUnderSampler (RUS): Thu gọn lớp đa số (Non-Fraud: 0) xuống một giới hạn an toàn.

SMOTE: Nội suy lớp thiểu số (Fraud: 1) lên mức cân bằng với lớp đa số vừa thu gọn.

Nguyên tắc tuyệt đối: Chỉ gọi module này trên tập Train. Tập Validation và Test phải giữ nguyên phân phối thực tế để đánh giá.

## 2. Hướng dẫn Tích hợp (Usage Guide)
Cách 1: Sử dụng hàm đóng gói (Functional Interface - Khuyên dùng)
Đây là cách nhanh nhất để tích hợp vào script huấn luyện. Mặc định hệ thống đang cấu hình under_sample_limit = 3,500,000. Nếu máy chủ huấn luyện gặp vấn đề về RAM, bạn cần chủ động hạ tham số này xuống (ví dụ: 150000 hoặc 300000).

# Import hàm từ file resampling.py (điều chỉnh đường dẫn import theo cấu trúc thư mục)
from resampling import apply_hybrid_resampling

# Giả định X_train, y_train đã được phân tách (Time-Series Split)
# Chạy pipeline xử lý
X_train_balanced, y_train_balanced = apply_hybrid_resampling(
    X_train=X_train, 
    y_train=y_train,
    under_sample_limit=3500000, # Tùy chỉnh giới hạn này dựa trên RAM của server train
    random_state=42
)

# Kiểm tra phân phối mới
print("Kích thước dữ liệu Train mới:", X_train_balanced.shape)
Cách 2: Sử dụng Class OOP (Tùy biến nâng cao)
Dành cho trường hợp bạn muốn định nghĩa file cấu hình riêng cho luồng Pipeline.

Python
from resampling import HybridResampler, ResamplingConfig

# Thiết lập tham số
config = ResamplingConfig(
    under_sample_limit=500000, 
    over_sample_strategy=1.0, # 1.0 tương đương tỷ lệ 1:1
    random_state=42
)

# Khởi tạo và thực thi
resampler = HybridResampler(config)
X_train_balanced, y_train_balanced = resampler.fit_resample(X_train, y_train)
3. Các hàm Hỗ trợ Thuật toán (Model Helpers)
Để tối ưu hóa các mô hình Tree-based (LightGBM, XGBoost, Random Forest), module cung cấp sẵn các hàm tính toán trọng số lớp. Bạn bơm trực tiếp kết quả của các hàm này vào siêu tham số (hyperparameters) của mô hình.

3.1. Dành cho XGBoost / LightGBM
Thuật toán Boosting sử dụng tham số scale_pos_weight để phạt lỗi trên lớp thiểu số.

Python
from resampling import get_scale_pos_weight
import lightgbm as lgb

# Tính toán trọng số dựa trên tập Train gốc
# LƯU Ý: Nếu đã dùng Hybrid Resampling tỷ lệ 1:1, bạn có thể bỏ qua bước này.
# Hàm này dùng khi bạn chạy các thực nghiệm KHÔNG dùng SMOTE.
weight = get_scale_pos_weight(y_train)

# Tích hợp vào mô hình
model = lgb.LGBMClassifier(
    scale_pos_weight=weight,
    random_state=42
)
3.2. Dành cho Random Forest / Logistic Regression
Các thuật toán thuộc Sklearn sử dụng tham số class_weight.

Python
from resampling import get_class_weight
from sklearn.ensemble import RandomForestClassifier

# Tính ma trận trọng số {0: w_0, 1: w_1}
weights_dict = get_class_weight(y_train)

# Tích hợp vào mô hình
model = RandomForestClassifier(
    class_weight=weights_dict,
    random_state=42
)