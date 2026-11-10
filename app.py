import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Dự đoán bệnh tim",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_models():
    svm = joblib.load('svm_benh_tim.pkl')
    km = None
    if os.path.exists('kmeans_benh_tim.pkl'):
        km = joblib.load('kmeans_benh_tim.pkl')
    return svm, km

try:
    model_svm, model_km = load_models()
except Exception as e:
    st.error(f"❌ Không tải được model: {e}")
    st.stop()

# ============================================================
# SIDEBAR — THÔNG TIN
# ============================================================
with st.sidebar:
    st.title("❤️ Dự đoán bệnh tim")
    st.markdown("---")
    st.markdown("**Nhóm 1** — Trí tuệ nhân tạo và ứng dụng")
    st.markdown("**GVHD:** Nguyễn Thị Thanh Hà")
    st.markdown("---")
    st.markdown("### 📌 Hướng dẫn sử dụng")
    st.markdown("""
    1. Nhập **13 chỉ số lâm sàng** ở các ô bên phải
    2. Nhấn nút **🔍 Dự đoán**
    3. Xem kết quả: **xác suất có bệnh tim**
    """)
    st.markdown("---")
    st.warning(
        "⚠️ **Lưu ý:** Đây là công cụ tham khảo, "
        "**không thay thế chẩn đoán của bác sĩ**."
    )

# ============================================================
# TIÊU ĐỀ CHÍNH
# ============================================================
st.title("❤️ Ứng dụng dự đoán nguy cơ bệnh tim")
st.caption("Mô hình: Support Vector Machine (SVM) — Huấn luyện trên bộ dữ liệu Heart Disease")
st.markdown("---")

# ============================================================
# FORM NHẬP LIỆU — 2 CỘT
# ============================================================
st.subheader("📋 Nhập thông tin bệnh nhân")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🧑 Thông tin cơ bản")

    age = st.number_input(
        "Tuổi (năm)",
        min_value=1, max_value=120, value=50, step=1,
        help="Tuổi của bệnh nhân"
    )

    sex = st.selectbox(
        "Giới tính",
        options=[1, 0],
        format_func=lambda x: "Nam" if x == 1 else "Nữ",
        help="Giới tính sinh học"
    )

    cp = st.selectbox(
        "Loại đau ngực (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Đau thắt ngực điển hình",
            1: "1 — Đau thắt ngực không điển hình",
            2: "2 — Đau không do thắt ngực",
            3: "3 — Không đau (asymptomatic)"
        }[x],
        help="Phân loại cơn đau ngực"
    )

    trestbps = st.slider(
        "Huyết áp tâm thu lúc nghỉ (mmHg)",
        min_value=80, max_value=220, value=120, step=1,
        help="Đo khi bệnh nhân nghỉ ngơi"
    )

    chol = st.slider(
        "Cholesterol huyết thanh (mg/dl)",
        min_value=100, max_value=600, value=200, step=5,
        help="Nồng độ cholesterol trong máu"
    )

    fbs = st.selectbox(
        "Đường huyết lúc đói > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "Có (> 120 mg/dl)" if x == 1 else "Không (≤ 120 mg/dl)",
        help="Đường huyết lúc đói có vượt 120 mg/dl không"
    )

with col2:
    st.markdown("##### 🩺 Kết quả xét nghiệm")

    restecg = st.selectbox(
        "Kết quả điện tâm đồ lúc nghỉ (restecg)",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Bình thường",
            1: "1 — Bất thường ST-T",
            2: "2 — Phì đại thất trái"
        }[x],
        help="Kết quả ECG lúc nghỉ"
    )

    thalach = st.slider(
        "Nhịp tim tối đa đạt được (bpm)",
        min_value=60, max_value=220, value=150, step=1,
        help="Nhịp tim cao nhất khi gắng sức"
    )

    exang = st.selectbox(
        "Đau thắt ngực khi gắng sức",
        options=[0, 1],
        format_func=lambda x: "Có" if x == 1 else "Không",
        help="Có xuất hiện đau thắt ngực khi gắng sức không"
    )

    oldpeak = st.slider(
        "ST chênh xuống do gắng sức (oldpeak)",
        min_value=0.0, max_value=7.0, value=1.0, step=0.1,
        help="Mức ST chênh xuống so với lúc nghỉ"
    )

    slope = st.selectbox(
        "Độ dốc đoạn ST (slope)",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Dốc xuống (downsloping)",
            1: "1 — Phẳng (flat)",
            2: "2 — Dốc lên (upsloping)"
        }[x],
        help="Độ dốc đoạn ST khi gắng sức"
    )

    ca = st.selectbox(
        "Số mạch chính bị tắc (ca)",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} mạch",
        help="Số mạch chính được tỏ màu qua fluoroscopy"
    )

    thal = st.selectbox(
        "Thalassemia (thal)",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Bình thường",
            2: "2 — Khuyết tật cố định (fixed defect)",
            3: "3 — Khuyết tật có thể phục hồi (reversible defect)"
        }[x],
        help="Kết quả xét nghiệm thalassemia"
    )

# ============================================================
# NÚT DỰ ĐOÁN
# ============================================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    du_doan = st.button("🔍  DỰ ĐOÁN", use_container_width=True, type="primary")

# ============================================================
# XỬ LÝ KẾT QUẢ
# ============================================================
if du_doan:
    # Tạo DataFrame input — đúng thứ tự 13 thuộc tính
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    # Dự đoán
    y_pred = model_svm.predict(input_df)[0]
    y_proba = model_svm.predict_proba(input_df)[0]
    proba_co_benh = y_proba[1]
    proba_khong_benh = y_proba[0]

    # ========================================================
    # HIỂN THỊ KẾT QUẢ
    # ========================================================
    st.markdown("---")
    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")

    # 2 cột: kết quả + xác suất
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if y_pred == 1:
            st.error("### ⚠️ CÓ NGUY CƠ BỆNH TIM")
            st.markdown(
                "Mô hình phát hiện các dấu hiệu **có khả năng mắc bệnh tim**. "
                "Bạn nên **đi khám bác sĩ chuyên khoa tim mạch** để kiểm tra kỹ hơn."
            )
        else:
            st.success("### ✅ KHÔNG PHÁT HIỆN BỆNH TIM")
            st.markdown(
                "Mô hình **không phát hiện dấu hiệu bệnh tim** từ các chỉ số đã nhập. "
                "Tuy nhiên, bạn vẫn nên **khám sức khỏe định kỳ** để theo dõi."
            )

    with res_col2:
        st.metric(
            label="🎯 Xác suất có bệnh tim",
            value=f"{proba_co_benh * 100:.2f}%",
            delta=f"{'Nguy cơ cao' if proba_co_benh > 0.5 else 'Nguy cơ thấp'}"
        )
        st.progress(float(proba_co_benh))

        st.metric(
            label="✅ Xác suất không bệnh",
            value=f"{proba_khong_benh * 100:.2f}%"
        )

    # ========================================================
    # PHÂN CỤM K-MEANS (nếu có)
    # ========================================================
    if model_km is not None:
        st.markdown("---")
        st.subheader("🔎 Phân cụm bệnh nhân (K-means)")

        scaler_km, kmeans = model_km
        features_km = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        X_km = input_df[features_km].values
        X_km_scaled = scaler_km.transform(X_km)
        cluster = int(kmeans.predict(X_km_scaled)[0])

        thong_tin_cum = {
            0: ("🟢 Cụm 0 — Nguy cơ thấp", "Nhóm lớn tuổi, nhịp tim thấp, oldpeak cao. Tỷ lệ mắc bệnh 25.88%."),
            1: ("🔴 Cụm 1 — Nguy cơ cao", "Nhóm trẻ, nhịp tim cao, oldpeak thấp. Tỷ lệ mắc bệnh 76.76%."),
            2: ("🟠 Cụm 2 — Nguy cơ trung bình", "Nhóm lớn tuổi, huyết áp và cholesterol cao. Tỷ lệ mắc bệnh 42.03%.")
        }
        ten, mo_ta = thong_tin_cum[cluster]
        st.info(f"**{ten}** — {mo_ta}")
        st.caption(
            "⚠️ K-means chỉ phân cụm dựa trên 5 chỉ số số (age, trestbps, chol, thalach, oldpeak), "
            "không dùng nhãn bệnh — chỉ mang tính tham khảo."
        )

    # ========================================================
    # BẢNG CHI TIẾT INPUT
    # ========================================================
    with st.expander("📋 Xem lại 13 chỉ số đã nhập"):
        hien_thi = input_df.T.reset_index()
        hien_thi.columns = ['Thuộc tính', 'Giá trị']
        st.dataframe(hien_thi, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "© 2026 — Nhóm 1 | Học phần: Trí tuệ nhân tạo và ứng dụng | "
    "GVHD: Nguyễn Thị Thanh Hà | "
    "Mô hình: SVM (kernel=linear, C=0.1) — Accuracy 85%, AUC 0.878"
)
