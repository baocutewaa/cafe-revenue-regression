import re

with open('scratch/generate_report_full.py', 'r', encoding='utf-8') as f:
    text = f.read()

TOC_ITEMS = [
    ("DANH SÁCH THÀNH VIÊN NHÓM 2", 1, True),
    ("MỤC LỤC", 1, True),
    ("DANH MỤC CÁC TỪ VIẾT TẮT VÀ KÝ HIỆU", 1, True),
    ("DANH MỤC BẢNG BIỂU VÀ HÌNH VẼ", 1, True),
    ("LỜI NÓI ĐẦU", 1, True),
    ("CHƯƠNG I: GIỚI THIỆU VỀ ĐỀ TÀI", 1, True),
    ("1.1. Lý do chọn đề tài", 2, False),
    ("1.2. Phạm vi và đối tượng nghiên cứu", 2, False),
    ("1.3. Mục tiêu của đề tài", 2, False),
    ("CHƯƠNG II: CƠ SỞ LÝ THUYẾT", 1, True),
    ("2.1. Tổng quan về bài toán học máy hồi quy chuỗi thời gian dạng bảng", 2, False),
    ("2.2. Các thuật toán học máy được nghiên cứu và ứng dụng", 2, False),
    ("2.2.1. Hồi quy tuyến tính có điều chuẩn (Ridge & Lasso Regression)", 3, False),
    ("2.2.2. Cây quyết định hồi quy (Decision Tree Regressor - CART)", 3, False),
    ("2.2.3. Rừng ngẫu nhiên (Random Forest) và Cây cực kỳ ngẫu nhiên (Extra Trees)", 3, False),
    ("2.2.4. Các thuật toán Gradient Boosting Machines (XGBoost, LightGBM, CatBoost)", 3, False),
    ("2.2.5. Các mô hình học máy truyền thống bổ trợ (KNN, SVR, MLP)", 3, False),
    ("2.3. Kiến trúc mô hình hóa phân tầng và khử xu thế (Hierarchical Base + Residual Learning)", 2, False),
    ("2.4. Phương pháp kết hợp mô hình Ensemble (Convex Weighted Blending)", 2, False),
    ("2.5. Các độ đo đánh giá mô hình hồi quy", 2, False),
    ("2.6. Chiến lược kiểm định chuỗi thời gian (Time-Based Validation Strategy)", 2, False),
    ("CHƯƠNG III: CÀI ĐẶT THỬ NGHIỆM MÔ HÌNH VÀ XÂY DỰNG ỨNG DỤNG TRÍ TUỆ NHÂN TẠO", 1, True),
    ("3.1. Thu thập dữ liệu và xử lý ghép bảng an toàn (Collect & Prepare)", 2, False),
    ("3.1.1. Phát hiện Rủi ro Dội dòng từ Bảng Holidays (Key Duplication Hazard)", 3, False),
    ("3.1.2. Giải pháp Thuật toán `prepare_holidays_table` và Ghép bảng An toàn", 3, False),
    ("3.2. Khám phá dữ liệu EDA và biên bản 10 lỗi dữ liệu bẩn (Explore & Data Anomalies)", 2, False),
    ("3.2.1. Sáu Quy luật Hành vi Kinh doanh F&B Cốt lõi (Domain Insights từ EDA)", 3, False),
    ("3.2.2. Hệ thống Trực quan hóa Khám phá Dữ liệu (EDA Visualizations)", 3, False),
    ("3.3. Tiền xử lý dữ liệu và thiết kế hệ thống 5 nhóm đặc trưng chuyên sâu (Preprocess)", 2, False),
    ("3.3.1. Thiết kế Hệ thống 5 Nhóm Đặc trưng Chuyên sâu (~54 Đặc trưng)", 3, False),
    ("3.3.2. Thuật toán `DeSeasonalizedStoreStats` và Xử lý Quán Mới (Cold-Start)", 3, False),
    ("3.4. Chiến lược phân chia dữ liệu theo thời gian (Time-Based Validation Split)", 2, False),
    ("3.4.1. Đánh giá Định lượng Mô hình Mốc chuẩn (Starter Baseline)", 3, False),
    ("3.5. Chuẩn hóa thang đo và đóng gói Pipeline chống rò rỉ dữ liệu (Scale & Pipeline)", 2, False),
    ("3.6. Huấn luyện 10 mô hình học máy và tối ưu hóa tổ hợp Ensemble (Model & Train)", 2, False),
    ("3.6.1. Tối ưu hóa Trọng số Lồi cho Mô hình Weighted Blend Ensemble", 3, False),
    ("3.6.2. Kiểm định Chéo Mở rộng theo Thời gian (4-Fold Expanding CV)", 3, False),
    ("3.7. Đánh giá chuyên sâu, phân tích bóc tách thành phần và chẩn đoán phần dư (Evaluate)", 2, False),
    ("3.7.1. Bảng Thực nghiệm Bóc tách Đóng góp Từng bước (Ablation Study Table)", 3, False),
    ("3.7.2. Chẩn đoán Phần dư và Độ chính xác Dự báo (Residual Diagnostics)", 3, False),
    ("3.7.3. Phân tích Sai số theo 5 Loại hình Cửa hàng (Store Type Breakdown)", 3, False),
    ("3.7.4. Tầm quan trọng Đặc trưng và Giải thích Mô hình (Feature Importance & SHAP)", 3, False),
    ("3.8. Triển khai ứng dụng thực tế và nộp bài Kaggle (Deploy & Optimization)", 2, False),
    ("3.8.1. Chiến lược Huấn luyện lại Toàn bộ (Full Retraining) và Tệp Nộp bài Kaggle", 3, False),
    ("3.8.2. Đóng gói Artifacts Duy nhất (`cafe_revenue_bundle.joblib`)", 3, False),
    ("3.8.3. Dịch vụ FastAPI Backend (`app/api_server.py`)", 3, False),
    ("3.8.4. Giao diện Streamlit Dashboard Quản trị (`app/streamlit_app.py`)", 3, False),
    ("3.8.5. Tối ưu hóa Chuyên sâu: Khử Xu thế Tăng trưởng Vĩ mô (Bước 9)", 3, False),
    ("CHƯƠNG IV: ĐÁNH GIÁ KẾT QUẢ NGHIÊN CỨU", 1, True),
    ("4.1. Kết quả đạt được", 2, False),
    ("4.1.1. Về mặt Học thuật và Mô hình hóa Học máy", 3, False),
    ("4.1.2. Về mặt Kỹ thuật Phần mềm và Độ tin cậy Hệ thống", 3, False),
    ("4.1.3. Giá trị Ứng dụng Thực tiễn trong Quản trị Chuỗi F&B", 3, False),
    ("4.1.4. Rèn luyện Kỹ năng Mềm và Phương pháp Nghiên cứu Khoa học", 3, False),
    ("4.2. Hạn chế của đề tài", 2, False),
    ("4.3. Hướng phát triển trong tương lai", 2, False),
    ("KẾT LUẬN", 1, True),
    ("TÀI LIỆU THAM KHẢO", 1, True),
    ("PHỤ LỤC", 1, True),
    ("PHỤ LỤC A: CẤU TRÚC THƯ MỤC MÃ NGUỒN DỰ ÁN", 2, False),
    ("PHỤ LỤC B: HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG", 2, False),
]

def norm(s):
    return re.sub(r'[\s`]+', ' ', s).strip().lower()

toc_set = {norm(t[0]): t for t in TOC_ITEMS}

pattern = re.compile(r'(add_h[123])\((f?["\'])(.*?)\2\)')
matches = pattern.findall(text)

unmatched = []
matched = []
front_matter_sub = [
    "1. danh mục ký hiệu toán học",
    "2. danh mục các từ viết tắt",
    "1. danh mục hình vẽ",
    "2. danh mục bảng biểu",
    "1. giáo trình và tài liệu học tập tiếng việt",
    "2. sách và bài báo khoa học quốc tế (tiếng anh)",
    "3. thư viện và tài liệu kỹ thuật trực tuyến"
]

for fn, _, title in matches:
    n = norm(title)
    if n in front_matter_sub:
        continue
    if n in toc_set:
        matched.append((fn, title))
    else:
        unmatched.append((fn, title))

print(f"Matched headings: {len(matched)} / {len(TOC_ITEMS)}")
if unmatched:
    print("Unmatched headings in code:")
    for u in unmatched:
        print(" ", u)
else:
    print("ALL HEADINGS MATCHED PERFECTLY!")
