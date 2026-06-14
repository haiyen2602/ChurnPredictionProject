import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap

from predict import (
    load_model_bundle,
    predict_single,
    predict_batch,
    get_risk_level,
    FEATURE_ORDER,
)

st.set_page_config(page_title="Banking Churn Prediction AI", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card { background: var(--background-color, #1a1f2e); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px 24px; text-align: center; margin-bottom: 8px; }
.metric-card .val { font-size:2rem; font-weight:700; margin:6px 0; }
.metric-card .lbl { font-size:0.75rem; opacity:.55; text-transform:uppercase; letter-spacing:1px; }
.result-box { border-radius: 16px; padding: 28px 24px; text-align: center; margin: 12px 0; }
.churn-box { background:rgba(232,89,60,0.12); border:2px solid #E8593C; }
.nochurn-box { background:rgba(29,158,117,0.12); border:2px solid #1D9E75; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-top: 10px; }
.sec-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; opacity: 0.5; margin-bottom: 6px; }
div[data-testid="stTabs"] button[role="tab"] { font-size : 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="⏳ Đang tải model AI...")
def load_model():
    try:
        return load_model_bundle('best_churn_model.pkl') + (None,)
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_model()

with st.sidebar:
    st.markdown("## 🏦 Banking Churn AI")
    st.caption("Dự đoán khách hàng rời bỏ ngân hàng (VIB Dataset)")
    st.divider()
    if model is not None:
        st.success(f"✅ Model sẵn sàng\n`{type(model).__name__}`")
    else:
        st.error("❌ Chưa load được model")
        if load_error: st.caption(load_error)

tab1, tab2, tab3 = st.tabs(["🔮 Dự đoán đơn lẻ", "📤 Batch Prediction", "📊 Model Dashboard"])

# TAB 1 — DỰ ĐOÁN ĐƠN LẺ
with tab1:
    if model is None: st.stop()
    st.markdown("### 🔮 Nhập thông tin khách hàng ngân hàng (28 Features)")

    with st.form("predict_form"):
        cols = st.columns(4)
        customer_raw = {}
        for i, feat in enumerate(FEATURE_ORDER):
            col = cols[i % 4]
            with col:
                # Xử lý input thông minh tuỳ theo tên field
                if feat in ['Client_gender', 'Staff_VIB', 'SMS', 'Verify_method', 'No_CC', 'No_DC']:
                    customer_raw[feat] = st.selectbox(feat, [0, 1])
                elif 'Amount' in feat or 'Balance' in feat:
                    customer_raw[feat] = st.number_input(feat, value=0.0, step=1000.0)
                else:
                    customer_raw[feat] = st.number_input(feat, value=0.0)
                    
        submitted = st.form_submit_button("🚀 Dự đoán ngay", use_container_width=True, type="primary")

    if submitted:
        prob, label, X_scaled, X_df = predict_single(model, scaler, customer_raw)
        emoji, risk_label, risk_color = get_risk_level(prob)

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            box_cls = "churn-box" if prob > 0.5 else "nochurn-box"
            st.markdown(f"""
            <div class="result-box {box_cls}">
                <div style="font-size:2.8rem">{emoji}</div>
                <div style="font-size:1.7rem;font-weight:700;margin:8px 0">{label}</div>
                <div style="font-size:2.4rem;font-weight:700;color:{risk_color}">{prob:.1%}</div>
                <div style="opacity:.65;font-size:0.9rem;margin-top:4px">Xác suất rời bỏ</div>
                <span class="badge" style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}66">{risk_label}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown("**💡 Khuyến nghị hành động:**")
            if prob >= 0.7:
                st.error("🚨 **Ưu tiên xử lý ngay** — Liên hệ trực tiếp, đề xuất ưu đãi giữ chân: lãi suất tốt hơn, miễn phí dịch vụ, tặng điểm thưởng hoặc nâng cấp gói Prenium.")
            elif prob >= 0.4:
                st.warning("⚡ **Theo dõi sát** — Gửi khảo sát hài lòng, giới thiệu thêm sản phẩm phù hợp.")
            else:
                st.success("✅ **Khách hàng ổn định** — Duy trì chất lượng dịch vụ, cross-sell thêm sản phẩm.")

        with right:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={'suffix': '%', 'font': {'size': 40}},
                gauge={'axis': {'range': [0, 100], 'tickwidth': 1}, 'bar': {'color': risk_color, 'thickness': 0.22},
                       'steps': [{'range': [0, 30], 'color': 'rgba(29,158,117,.15)'}, {'range': [30, 60], 'color': 'rgba(243,156,18,.15)'}, {'range': [60,100], 'color': 'rgba(232,89,60,.15)'}],
                       'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.82, 'value': 50}},
                title={'text': "Churn Probability", 'font': {'size': 15}},
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()
        st.markdown("#### 🔍 Feature Importance (Rút trích tự động từ Model)")
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            fi_df = pd.DataFrame({'Feature': FEATURE_ORDER, 'Importance': imp}).sort_values('Importance').tail(12)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
            fig_fi.update_layout(height=380, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_fi, use_container_width=True)


# TAB 2 — BATCH PREDICTION
with tab2:
    if model is None: st.stop()
    st.markdown("### 📤 Dự đoán hàng loạt từ file CSV")
    
    sample_row = {f: 0 for f in FEATURE_ORDER}
    sample_row['Customer_number'] = 1
    template_df = pd.DataFrame([sample_row])

    _, br = st.columns([3, 1])
    with br:
        st.download_button("⬇️ Tải file CSV mẫu", data=template_df.to_csv(index=False).encode('utf-8'), file_name="template_bank_churn.csv", mime="text/csv")

    uploaded = st.file_uploader("📁 Upload file CSV", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.success(f"✅ Đã tải **{len(df_up):,}** khách hàng")

        if st.button("🚀 Chạy dự đoán batch", type="primary"):
            with st.spinner(f"⏳ Đang xử lý {len(df_up):,} khách hàng..."):
                result_df = predict_batch(model, scaler, df_up)

            n_total = len(result_df)
            n_churn = (result_df['Prediction'] == 'CHURN').sum()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng KH", f"{n_total:,}")
            m2.metric("Dự đoán CHURN", f"{n_churn:,}")
            m3.metric("Tỉ lệ Churn", f"{n_churn/n_total:.1%}")
            
            st.dataframe(result_df.head(50), use_container_width=True)

# TAB 3 — MODEL DASHBOARD
with tab3:
    st.markdown("### 📊 Thông tin & Hiệu suất")
    st.info("Mô hình được huấn luyện trên kiến trúc tập dữ liệu VIB - 66,736 mẫu, sử dụng XGBoost sau khi lấy mẫu SMOTE để khắc phục tình trạng Imbalanced Data (Tỉ lệ churn 35%).")