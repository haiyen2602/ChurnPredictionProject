"""
app.py — Streamlit Web App | Customer Churn Prediction
Chạy local : streamlit run app.py
Deploy     : push lên GitHub → Streamlit Cloud
"""

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
    get_vi_feature_names,
    ENCODING_MAP,
    FEATURE_ORDER,
    FEATURE_LABELS,
)

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Churn Prediction AI",
    page_icon   = "🎯",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Card metric */
.metric-card {
    background  : var(--background-color, #1a1f2e);
    border      : 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding     : 20px 24px;
    text-align  : center;
    margin-bottom: 8px;
}
.metric-card .val  { font-size:2rem; font-weight:700; margin:6px 0; }
.metric-card .lbl  { font-size:0.75rem; opacity:.55; text-transform:uppercase; letter-spacing:1px; }

/* Result box */
.result-box {
    border-radius: 16px;
    padding      : 28px 24px;
    text-align   : center;
    margin       : 12px 0;
}
.churn-box    { background:rgba(232,89,60,0.12); border:2px solid #E8593C; }
.nochurn-box  { background:rgba(29,158,117,0.12); border:2px solid #1D9E75; }

/* Risk badge */
.badge {
    display      : inline-block;
    padding      : 4px 14px;
    border-radius: 20px;
    font-size    : 0.85rem;
    font-weight  : 600;
    margin-top   : 10px;
}

/* Section divider label */
.sec-label {
    font-size    : 0.8rem;
    font-weight  : 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity      : 0.55;
    margin-bottom: 6px;
}

/* Tab font */
div[data-testid="stTabs"] button[role="tab"] {
    font-size : 1rem;
    font-weight: 600;
}

/* Tighten sidebar */
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (cache — chỉ load 1 lần)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải model AI...")
def load_model():
    try:
        model, scaler = load_model_bundle('best_churn_model.pkl')
        return model, scaler, None
    except FileNotFoundError:
        return None, None, "Không tìm thấy file `best_churn_model.pkl`."
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_model()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Churn Prediction AI")
    st.caption("Dự đoán khách hàng rời bỏ dịch vụ viễn thông")
    st.divider()

    if model is not None:
        st.success(f"✅ Model sẵn sàng  \n`{type(model).__name__}`")
    else:
        st.error("❌ Chưa load được model")
        if load_error:
            st.caption(load_error)
        st.info(
            "**Cách tạo file model:**  \n"
            "1. Chạy notebook Colab đến cuối  \n"
            "2. Tải `best_churn_model.pkl`  \n"
            "3. Đặt vào cùng thư mục với `app.py`"
        )

    st.divider()
    st.markdown("""
**3 tính năng chính:**
- 🔮 **Đơn lẻ** — Nhập thông tin 1 KH, xem kết quả + giải thích SHAP
- 📤 **Batch** — Upload CSV, dự đoán & xuất kết quả
- 📊 **Dashboard** — Xem hiệu suất mô hình
    """)
    st.divider()
    st.caption("📘 Project cuối kỳ AI  \nDataset: IBM Telco Customer Churn")


# ─────────────────────────────────────────────────────────────
# 3 TABS CHÍNH
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮  Dự đoán đơn lẻ",
    "📤  Batch Prediction",
    "📊  Model Dashboard",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — DỰ ĐOÁN ĐƠN LẺ
# ══════════════════════════════════════════════════════════════
with tab1:
    if model is None:
        st.warning("⚠️ Cần file `best_churn_model.pkl`. Xem hướng dẫn ở sidebar.")
        st.stop()

    st.markdown("### 🔮 Nhập thông tin khách hàng")

    # ── FORM INPUT ──────────────────────────────────────────
    with st.form("predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        # Cột 1 — Thông tin cá nhân
        with col1:
            st.markdown('<p class="sec-label">👤 Thông tin cá nhân</p>', unsafe_allow_html=True)
            gender     = st.selectbox("Giới tính", ["Male", "Female"])
            senior     = st.selectbox("Người cao tuổi (≥65)", [0, 1],
                                      format_func=lambda x: "Có" if x else "Không")
            partner    = st.selectbox("Có đối tác", ["No", "Yes"],
                                      format_func=lambda x: "Có" if x == "Yes" else "Không")
            dependents = st.selectbox("Có người phụ thuộc", ["No", "Yes"],
                                      format_func=lambda x: "Có" if x == "Yes" else "Không")
            tenure     = st.slider("Số tháng sử dụng", 0, 72, 12,
                                   help="Khách hàng đã dùng dịch vụ bao nhiêu tháng")

        # Cột 2 — Dịch vụ
        with col2:
            st.markdown('<p class="sec-label">📡 Dịch vụ đang dùng</p>', unsafe_allow_html=True)
            phone_svc  = st.selectbox("Dịch vụ điện thoại",  ["Yes", "No"])
            multi_line = st.selectbox("Nhiều đường dây",      ["No", "Yes", "No phone service"])
            internet   = st.selectbox("Dịch vụ Internet",     ["Fiber optic", "DSL", "No"])
            security   = st.selectbox("Bảo mật Online",       ["No", "Yes", "No internet service"])
            backup     = st.selectbox("Sao lưu Online",       ["No", "Yes", "No internet service"])
            protection = st.selectbox("Bảo vệ thiết bị",     ["No", "Yes", "No internet service"])
            tech_sup   = st.selectbox("Hỗ trợ kỹ thuật",     ["No", "Yes", "No internet service"])
            stream_tv  = st.selectbox("Xem TV trực tuyến",   ["No", "Yes", "No internet service"])
            stream_mv  = st.selectbox("Xem phim trực tuyến", ["No", "Yes", "No internet service"])

        # Cột 3 — Hợp đồng & Tài chính
        with col3:
            st.markdown('<p class="sec-label">💳 Hợp đồng & Thanh toán</p>', unsafe_allow_html=True)
            contract   = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
            paperless  = st.selectbox("Hóa đơn điện tử", ["Yes", "No"])
            payment    = st.selectbox("Phương thức thanh toán", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ])
            monthly    = st.number_input("Chi phí hàng tháng ($)", 0.0, 200.0, 65.0, step=0.5)
            default_total = round(monthly * max(tenure, 1), 1)
            total      = st.number_input("Tổng chi phí ($)", 0.0, 10000.0,
                                         float(default_total), step=1.0)

        submitted = st.form_submit_button(
            "🚀  Dự đoán ngay",
            use_container_width=True,
            type="primary",
        )

    # ── KẾT QUẢ ─────────────────────────────────────────────
    if submitted:
        customer_raw = {
            'gender': gender, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_svc, 'MultipleLines': multi_line,
            'InternetService': internet, 'OnlineSecurity': security,
            'OnlineBackup': backup, 'DeviceProtection': protection,
            'TechSupport': tech_sup, 'StreamingTV': stream_tv,
            'StreamingMovies': stream_mv, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total,
        }

        prob, label, X_scaled, X_df = predict_single(model, scaler, customer_raw)
        emoji, risk_label, risk_color = get_risk_level(prob)

        st.divider()
        left, right = st.columns([1, 1])

        # Kết quả + khuyến nghị
        with left:
            box_cls = "churn-box" if prob > 0.5 else "nochurn-box"
            st.markdown(f"""
            <div class="result-box {box_cls}">
                <div style="font-size:2.8rem">{emoji}</div>
                <div style="font-size:1.7rem;font-weight:700;margin:8px 0">{label}</div>
                <div style="font-size:2.4rem;font-weight:700;color:{risk_color}">{prob:.1%}</div>
                <div style="opacity:.65;font-size:0.9rem;margin-top:4px">xác suất rời bỏ dịch vụ</div>
                <span class="badge" style="background:{risk_color}22;
                      color:{risk_color};border:1px solid {risk_color}66">
                    {risk_label}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**💡 Khuyến nghị hành động:**")
            if prob >= 0.7:
                st.error(
                    "🚨 **Ưu tiên xử lý ngay** — Liên hệ trực tiếp, đề xuất "
                    "ưu đãi giữ chân: giảm phí, nâng cấp gói, tặng tháng miễn phí."
                )
            elif prob >= 0.4:
                st.warning(
                    "⚡ **Theo dõi sát** — Gửi khảo sát hài lòng, "
                    "offer thêm dịch vụ (TechSupport, OnlineSecurity)."
                )
            else:
                st.success(
                    "✅ **Khách hàng ổn định** — Duy trì chất lượng dịch vụ, "
                    "có thể upsell gói cao hơn hoặc thêm dịch vụ streaming."
                )

            # Thông tin tóm tắt
            with st.expander("📋 Thông tin đã nhập"):
                summary = {
                    "Giới tính": gender,
                    "Số tháng": tenure,
                    "Hợp đồng": contract,
                    "Internet": internet,
                    "Chi phí/tháng": f"${monthly:.1f}",
                    "Tổng chi phí": f"${total:.1f}",
                }
                for k, v in summary.items():
                    st.text(f"  {k}: {v}")

        # Gauge chart
        with right:
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                number= {'suffix': '%', 'font': {'size': 40}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'gray'},
                    'bar' : {'color': risk_color, 'thickness': 0.22},
                    'steps': [
                        {'range': [0,  30], 'color': 'rgba(29,158,117,.15)'},
                        {'range': [30, 60], 'color': 'rgba(243,156,18,.15)'},
                        {'range': [60,100], 'color': 'rgba(232,89,60,.15)'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 3},
                        'thickness': 0.82,
                        'value': 50,
                    },
                },
                title={'text': "Churn Probability", 'font': {'size': 15}},
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=60, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature summary nhanh
            st.markdown("**📌 Features chính:**")
            feat_quick = {
                "Hợp đồng":      contract,
                "Số tháng":      f"{tenure} tháng",
                "Chi phí/tháng": f"${monthly:.0f}",
                "Số dịch vụ":    int(X_df['NumServices'].iloc[0]),
            }
            q1, q2 = st.columns(2)
            items = list(feat_quick.items())
            for i, (k, v) in enumerate(items):
                (q1 if i < 2 else q2).metric(k, v)

        # ── SHAP WATERFALL ────────────────────────────────
        st.divider()
        st.markdown("#### 🔍 Giải thích dự đoán — SHAP Waterfall")
        st.caption(
            "Mỗi thanh cho biết một feature đẩy xác suất churn **lên** (đỏ) "
            "hay **xuống** (xanh) so với mức trung bình của tập train."
        )

        try:
            feat_names_vi = get_vi_feature_names(list(X_df.columns))
            explainer     = shap.TreeExplainer(model)
            shap_vals     = explainer.shap_values(X_df)

            fig_w, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            shap.waterfall_plot(
                shap.Explanation(
                    values      = shap_vals[0],
                    base_values = explainer.expected_value,
                    data        = X_df.iloc[0].values,
                    feature_names = feat_names_vi,
                ),
                max_display=12,
                show=False,
            )
            plt.tight_layout()
            st.pyplot(fig_w, use_container_width=True)
            plt.close()

        except Exception:
            # Fallback: feature importance thông thường
            st.info("Model này không hỗ trợ SHAP TreeExplainer. Hiển thị Feature Importance.")
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                fi_df = pd.DataFrame({
                    'Feature':    get_vi_feature_names(FEATURE_ORDER),
                    'Importance': imp,
                }).sort_values('Importance').tail(15)

                fig_fi = px.bar(
                    fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues',
                    title="Feature Importance (XGBoost)",
                )
                fig_fi.update_layout(
                    height=420, showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                )
                st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        st.warning("⚠️ Cần file `best_churn_model.pkl`. Xem hướng dẫn ở sidebar.")
        st.stop()

    st.markdown("### 📤 Dự đoán hàng loạt từ file CSV")
    st.caption("Upload file CSV cùng cấu trúc Telco dataset — hệ thống tự predict toàn bộ.")

    # Template download
    TEMPLATE_COLS = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
    ]
    sample_row = {
        'customerID': 'CUST-001', 'gender': 'Male', 'SeniorCitizen': 0,
        'Partner': 'No', 'Dependents': 'No', 'tenure': 2,
        'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.5, 'TotalCharges': 171.0,
    }
    template_df = pd.DataFrame([sample_row])

    bl, br = st.columns([3, 1])
    with br:
        st.download_button(
            "⬇️ Tải file CSV mẫu",
            data      = template_df.to_csv(index=False).encode('utf-8'),
            file_name = "template_churn.csv",
            mime      = "text/csv",
        )

    uploaded = st.file_uploader("📁 Upload file CSV khách hàng", type=["csv"])

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        n_rows = len(df_up)
        st.success(f"✅ Đã tải **{n_rows:,}** khách hàng")

        with st.expander("👁️ Xem trước 5 dòng đầu"):
            st.dataframe(df_up.head(), use_container_width=True)

        if st.button("🚀 Chạy dự đoán batch", type="primary"):
            with st.spinner(f"⏳ Đang xử lý {n_rows:,} khách hàng..."):
                result_df = predict_batch(model, scaler, df_up)

            # ── METRICS ──────────────────────────────────
            n_churn    = (result_df['Prediction'] == 'CHURN').sum()
            churn_rate = n_churn / n_rows
            high_risk  = (result_df['Risk_Level'] == '🔴 Cao').sum()
            avg_prob   = result_df['Churn_Probability'].mean()

            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                ("Tổng khách hàng", f"{n_rows:,}",   "#3266ad"),
                ("Dự đoán CHURN",   f"{n_churn:,}",  "#E8593C"),
                ("Tỉ lệ churn",     f"{churn_rate:.1%}", "#f39c12"),
                ("Nguy cơ cao",     f"{high_risk:,}", "#E8593C"),
            ]
            for col, (lbl, val, color) in zip([m1, m2, m3, m4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="lbl">{lbl}</div>
                        <div class="val" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("")

            # ── CHARTS ───────────────────────────────────
            ch1, ch2 = st.columns(2)

            with ch1:
                risk_cnt = result_df['Risk_Level'].value_counts()
                fig_pie  = go.Figure(go.Pie(
                    labels         = risk_cnt.index.tolist(),
                    values         = risk_cnt.values.tolist(),
                    hole           = 0.44,
                    marker_colors  = ['#1D9E75', '#f39c12', '#E8593C'],
                    textinfo       = 'label+percent',
                ))
                fig_pie.update_layout(
                    title           = "Phân bố mức độ nguy cơ",
                    height          = 320,
                    paper_bgcolor   = 'rgba(0,0,0,0)',
                    font_color      = 'white',
                    showlegend      = False,
                    margin          = dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch2:
                fig_hist = px.histogram(
                    result_df, x='Churn_Probability', nbins=30,
                    color='Prediction',
                    color_discrete_map={
                        'CHURN': '#E8593C',
                        'KHÔNG CHURN': '#1D9E75',
                    },
                    title  = "Phân phối xác suất Churn",
                    labels = {'Churn_Probability': 'Xác suất', 'count': 'Số KH'},
                )
                fig_hist.update_layout(
                    height        = 320,
                    paper_bgcolor = 'rgba(0,0,0,0)',
                    plot_bgcolor  = 'rgba(0,0,0,0)',
                    font_color    = 'white',
                    legend_title  = "",
                    margin        = dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── BẢNG KẾT QUẢ ─────────────────────────────
            st.markdown("#### 📋 Kết quả chi tiết (top 50 nguy cơ cao nhất)")

            id_col  = 'customerID' if 'customerID' in result_df.columns else None
            base_cols = ['Churn_Probability', 'Prediction', 'Risk_Level',
                         'tenure', 'Contract', 'MonthlyCharges']
            show_cols = ([id_col] + base_cols) if id_col else base_cols

            st.dataframe(
                result_df[show_cols].head(50)
                    .style
                    .background_gradient(
                        subset=['Churn_Probability'],
                        cmap='RdYlGn_r', vmin=0, vmax=1,
                    )
                    .format({'Churn_Probability': '{:.1%}'}),
                use_container_width=True,
                height=400,
            )

            # ── DOWNLOAD ─────────────────────────────────
            st.download_button(
                "⬇️ Tải kết quả đầy đủ (CSV)",
                data      = result_df.to_csv(index=False).encode('utf-8'),
                file_name = "churn_predictions.csv",
                mime      = "text/csv",
            )


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Thông tin & Hiệu suất Mô hình")

    # ── PIPELINE & DATASET ───────────────────────────────
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("#### 🏗️ Pipeline xử lý")
        pipeline_data = {
            "Bước": [
                "1. Load data",
                "2. Xử lý missing",
                "3. Encode categorical",
                "4. Feature engineering",
                "5. Scale features",
                "6. Cân bằng class",
                "7. Train models",
                "8. Ensemble",
                "9. Explainability",
            ],
            "Kỹ thuật": [
                "IBM Telco CSV → DataFrame",
                "Median fill (TotalCharges)",
                "LabelEncoder (15 cột)",
                "ChargePerTenure, NumServices",
                "StandardScaler",
                "SMOTE (oversampling)",
                "LR · DT · RF · XGB · SVM",
                "Soft VotingClassifier",
                "SHAP TreeExplainer",
            ],
        }
        st.dataframe(
            pd.DataFrame(pipeline_data),
            use_container_width=True,
            hide_index=True,
        )

    with d2:
        st.markdown("#### 📐 Thông số Dataset")
        dataset_data = {
            "Thông số": [
                "Dataset", "Tổng mẫu", "Features gốc",
                "Features sau engineer", "Tỉ lệ Churn",
                "Train / Test split", "Cross-validation",
            ],
            "Giá trị": [
                "IBM Telco Customer Churn", "7.043 khách hàng", "21 cột",
                "23 cột (+ChargePerTenure, NumServices)", "26.5%",
                "80% / 20% (stratified)", "StratifiedKFold (k=5)",
            ],
        }
        st.dataframe(
            pd.DataFrame(dataset_data),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── PERFORMANCE CHART ────────────────────────────────
    st.markdown("#### 🎯 So sánh AUC-ROC các mô hình")
    st.caption("Số liệu từ kết quả chạy trên tập test (20%). Cập nhật bằng cách chạy lại notebook.")

    perf = {
        'Model':    ['Logistic Regression', 'Decision Tree', 'Random Forest',
                     'XGBoost', 'SVM', '🏆 Soft Voting'],
        'AUC-ROC': [0.843, 0.761, 0.847, 0.861, 0.849, 0.865],
        'F1-Score': [0.598, 0.531, 0.614, 0.629, 0.608, 0.633],
        'Recall':   [0.551, 0.502, 0.571, 0.598, 0.571, 0.602],
        'Precision':[0.655, 0.563, 0.668, 0.664, 0.653, 0.669],
    }
    perf_df = pd.DataFrame(perf)

    chart_metric = st.selectbox(
        "Chọn metric:",
        ['AUC-ROC', 'F1-Score', 'Recall', 'Precision'],
        index=0,
    )

    colors = [
        '#ffd700' if '🏆' in m else '#3266ad'
        for m in perf_df['Model']
    ]
    fig_bar = go.Figure(go.Bar(
        x           = perf_df['Model'],
        y           = perf_df[chart_metric],
        marker_color= colors,
        text        = perf_df[chart_metric].round(3),
        textposition= 'outside',
    ))
    fig_bar.update_layout(
        yaxis_range   = [
            perf_df[chart_metric].min() - 0.05,
            perf_df[chart_metric].max() + 0.05,
        ],
        height        = 380,
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font_color    = 'white',
        xaxis_tickangle = -20,
        showlegend    = False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── FULL METRICS TABLE ────────────────────────────────
    st.markdown("#### 📋 Bảng kết quả đầy đủ")
    styled_perf = (
        perf_df.set_index('Model')
            .style
            .highlight_max(axis=0, color='rgba(29,158,117,0.3)')
            .format('{:.4f}')
    )
    st.dataframe(styled_perf, use_container_width=True)

    st.divider()

    # ── FEATURE IMPORTANCE ────────────────────────────────
    st.markdown("#### 🔍 Feature Importance (XGBoost — ước tính)")
    fi_data = {
        'Feature':    [
            'Loại hợp đồng', 'Chi phí tháng', 'Số tháng dùng',
            'Chi phí/Tenure', 'Dịch vụ Internet', 'Tổng chi phí',
            'Hóa đơn ĐT', 'Phương thức TT', 'Hỗ trợ KT', 'Bảo mật Online',
            'Số dịch vụ', 'Sao lưu Online',
        ],
        'Importance': [
            0.182, 0.156, 0.143, 0.098, 0.087, 0.071,
            0.063, 0.058, 0.047, 0.039, 0.032, 0.024,
        ],
    }
    fi_df = pd.DataFrame(fi_data).sort_values('Importance')
    fig_fi = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        title="Top 12 Feature Importance (XGBoost)",
    )
    fig_fi.update_layout(
        height        = 420,
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font_color    = 'white',
        showlegend    = False,
        coloraxis_showscale = False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info(
        "**Lưu ý:** Các chỉ số AUC-ROC và Feature Importance trên đây là "
        "giá trị **ước tính** dựa trên kết quả chạy notebook. "
        "Để cập nhật số liệu chính xác từ model thực tế, export thêm file "
        "`results.json` từ notebook và load vào đây."
    )
