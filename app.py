import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# 1. Cấu hình trang web
st.set_page_config(page_title="Dự báo Churn", layout="wide", page_icon="📊")

# Thông tin sinh viên thực hiện đồ án
st.markdown("<h1 style='text-align: center;'>Hệ thống Dự báo Khách hàng rời bỏ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Phát triển bởi: <b>Nguyễn Hải Yến</b> (MSSV: 23070039)</p>", unsafe_allow_html=True)
st.divider()

# 2. Tải các thành phần đã đóng gói từ Colab
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except Exception as e:
    st.error("Lỗi tải mô hình. Vui lòng kiểm tra lại các file .pkl đã có trong thư mục chưa.")
    st.stop()

# 3. Giao diện nhập liệu từ người dùng
st.subheader("Nhập thông tin khách hàng")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Điểm tín dụng", min_value=300, max_value=850, value=650)
    age = st.number_input("Độ tuổi", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    geography = st.selectbox("Quốc gia", ["Pháp", "Đức", "Tây Ban Nha"])

with col2:
    tenure = st.number_input("Số năm gắn bó", min_value=0, max_value=10, value=5)
    balance = st.number_input("Số dư tài khoản ($)", min_value=0.0, value=50000.0)
    num_of_products = st.selectbox("Số lượng sản phẩm sử dụng", [1, 2, 3, 4])

with col3:
    has_crcard = st.selectbox("Sở hữu thẻ tín dụng?", ["Có", "Không"])
    is_active_member = st.selectbox("Là thành viên tích cực?", ["Có", "Không"])
    estimated_salary = st.number_input("Lương ước tính ($)", min_value=0.0, value=60000.0)

# 4. Tiền xử lý dữ liệu đầu vào để khớp với cấu trúc lúc Train
def prepare_input_data():
    # Tạo một từ điển với tất cả các cột mặc định là 0
    input_dict = {col: 0 for col in features}
    
    # Gán các giá trị dạng số
    input_dict['CreditScore'] = credit_score
    input_dict['Age'] = age
    input_dict['Tenure'] = tenure
    input_dict['Balance'] = balance
    input_dict['NumOfProducts'] = num_of_products
    input_dict['EstimatedSalary'] = estimated_salary
    
    # Mã hóa Label Encoding (Giả định Nam = 1, Nữ = 0 từ Colab)
    input_dict['Gender'] = 1 if gender == "Nam" else 0
    
    # Mã hóa nhị phân
    input_dict['HasCrCard'] = 1 if has_crcard == "Có" else 0
    input_dict['IsActiveMember'] = 1 if is_active_member == "Có" else 0
    
    # Mã hóa One-Hot Encoding cho Quốc gia (Pháp bị drop_first)
    if geography == "Đức" and 'Geography_Germany' in features:
        input_dict['Geography_Germany'] = 1
    elif geography == "Tây Ban Nha" and 'Geography_Spain' in features:
        input_dict['Geography_Spain'] = 1

    # Chuyển thành DataFrame và sắp xếp đúng thứ tự cột
    df = pd.DataFrame([input_dict])
    df = df[features]
    return df

# 5. Hàm gọi API Gemini để đưa ra chiến lược
def get_prenium_retention_strategy(data, prob):
    # Cấu hình API Key (Lấy tại Google AI Studio)
    genai.configure(api_key="NHẬP_API_KEY_CỦA_BẠN_VÀO_ĐÂY") 
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Đóng vai trò là chuyên gia phân tích dữ liệu ngân hàng. 
    Khách hàng này có nguy cơ rời bỏ dịch vụ rất cao (Xác suất: {prob:.2%}).
    Đặc điểm: {age} tuổi, giới tính {gender}, số dư: ${balance}, lương: ${estimated_salary}, dùng {num_of_products} sản phẩm.
    Hãy đề xuất 3 hành động cụ thể, ngắn gọn (mỗi hành động 1 gạch đầu dòng) để giữ chân khách hàng này.
    """
    try:
        response = ai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Hệ thống AI tạm thời không khả dụng để đưa ra đề xuất."

# 6. Xử lý dự đoán
st.divider()
if st.button("🚀 Thực hiện Dự báo", type="primary", use_container_width=True):
    # Chuẩn bị và scale dữ liệu
    input_df = prepare_input_data()
    scaled_data = scaler.transform(input_df)
    
    # Gọi mô hình dự đoán
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Hiển thị kết quả
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        if prediction == 1:
            st.error(f"⚠️ CẢNH BÁO: Rủi ro rời bỏ cao!\n\n**Xác suất: {probability:.2%}**")
        else:
            st.success(f"✅ AN TOÀN: Khách hàng có xu hướng ở lại.\n\n**Xác suất rời bỏ: {probability:.2%}**")
            
    with col_res2:
        if prediction == 1:
            with st.spinner("Đang trích xuất tư vấn từ Prenium AI Insights..."):
                strategy = get_prenium_retention_strategy(input_df, probability)
                st.info(f"💡 **Prenium AI Insights - Đề xuất chiến lược:**\n\n{strategy}")
