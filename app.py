# Ứng dụng dự đoán bệnh tim — chạy: streamlit run app.py
import streamlit as st, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

@st.cache_resource
def load():
    d = pd.read_excel('Nhom1_BenhTim.xlsx'); X, y = d.drop(columns='target'), d.target
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42).fit(X, y)
    sv = make_pipeline(StandardScaler(), SVC(C=0.1, kernel='linear', probability=True, random_state=42)).fit(X, y)
    return X, rf, sv
X, rf, sv = load()
st.title('Dự đoán nguy cơ bệnh tim')
st.caption('Công cụ học thuật, không thay thế chẩn đoán y khoa.')
lab = {'age':'Tuổi','sex':'Giới tính (1=Nam, 0=Nữ)','cp':'Loại đau ngực (0-3)','trestbps':'Huyết áp nghỉ','chol':'Cholesterol',
 'fbs':'Đường huyết đói >120 (1/0)','restecg':'ECG nghỉ (0-2)','thalach':'Nhịp tim tối đa','exang':'Đau thắt ngực khi gắng sức (1/0)',
 'oldpeak':'ST chênh xuống (oldpeak)','slope':'Độ dốc ST (0-2)','ca':'Số mạch chính (0-3)','thal':'Thalassemia (1-3)'}
vals = {c: st.number_input(lab[c], float(X[c].min()), float(X[c].max()), float(X[c].median())) for c in X.columns}
name = st.radio('Mô hình', ['SVM', 'Random Forest'], horizontal=True)
if st.button('Dự đoán'):
    row = pd.DataFrame([vals]); p = (sv if name == 'SVM' else rf).predict_proba(row)[0, 1]
    st.metric('Xác suất có bệnh tim', f'{p:.1%}'); (st.error if p >= .5 else st.success)('Nguy cơ cao' if p >= .5 else 'Nguy cơ thấp')
