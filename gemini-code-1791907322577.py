import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Hiển thị tiêu đề
st.title("QUẢN LÝ ĐIỂM SINH VIÊN")

# Khởi tạo dữ liệu
data = {
    'Họ và tên': ['An', 'Bình', 'Chi', 'Dũng', 'Hà', 'Lan', 'Minh', 'Nam', 'Phúc', 'Trang'],
    'Chuyên cần': [8, 9, 7, 10, 8, 6, 9, 8, 7, 10],
    'Giữa kỳ': [7.5, 8, 6.5, 9, 7, 5, 8.5, 7, 6.5, 9],
    'Cuối kỳ': [8, 8.5, 6, 9.5, 7.5, 6, 8.5, 6.5, 8, 9.5]
}
df = pd.DataFrame(data)
df['Tong_ket'] = 0.2 * df['Chuyên cần'] + 0.3 * df['Giữa kỳ'] + 0.5 * df['Cuối kỳ']

def xep_loai(diem):
    if diem >= 8.5: return 'Giỏi'
    elif 7.0 <= diem < 8.5: return 'Khá'
    elif 5.0 <= diem < 7.0: return 'Trung bình'
    else: return 'Yếu'

df['Xep_loai'] = df['Tong_ket'].apply(xep_loai)

# 2. Hiển thị bảng điểm
st.subheader("Bảng điểm chi tiết")
st.dataframe(df)

# 3. Hiển thị thống kê
st.subheader("Thống kê")
st.write(f"**Điểm trung bình của lớp:** {df['Tong_ket'].mean():.2f}")
st.write(f"**Sinh viên điểm cao nhất:** {df.loc[df['Tong_ket'].idxmax()]['Họ và tên']} ({df['Tong_ket'].max()})")
st.write(f"**Sinh viên điểm thấp nhất:** {df.loc[df['Tong_ket'].idxmin()]['Họ và tên']} ({df['Tong_ket'].min()})")
st.write(f"**Số sinh viên đạt:** {(df['Tong_ket'] >= 5.0).sum()}")

# 4. Tạo Selectbox chọn sinh viên
st.subheader("Tra cứu thông tin sinh viên")
sv_chon = st.selectbox("Chọn sinh viên", df['Họ và tên'])
thong_tin_sv = df[df['Họ và tên'] == sv_chon].iloc[0]
st.write(f"**Điểm chuyên cần:** {thong_tin_sv['Chuyên cần']}")
st.write(f"**Điểm giữa kỳ:** {thong_tin_sv['Giữa kỳ']}")
st.write(f"**Điểm cuối kỳ:** {thong_tin_sv['Cuối kỳ']}")
st.write(f"**Điểm tổng kết:** {thong_tin_sv['Tong_ket']}")
st.write(f"**Xếp loại:** {thong_tin_sv['Xep_loai']}")

# 5. Hiển thị biểu đồ
st.subheader("Biểu đồ điểm tổng kết")
fig, ax = plt.subplots()
ax.barh(df['Họ và tên'], df['Tong_ket'], color='cornflowerblue')
ax.set_xlabel('Điểm tổng kết')
ax.set_ylabel('Họ và tên')
st.pyplot(fig)

# 6. Thông tin người tạo (Footer)
st.markdown("<br><hr><p style='font-size: 12px; text-align: center;'>Người tạo: [Tên của bạn] - MSSV: [MSSV của bạn]</p>", unsafe_allow_html=True)