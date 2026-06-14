"""
app.py — Streamlit Web App | Banking Customer Churn Prediction
Dataset: trnhuytun/churn-prediction-dataset (Kaggle)
Chạy  : streamlit run app.py
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
    page_title  = "Banking Churn Prediction AI",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
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

.result-box {
    border-radius: 16px;
    padding      : 28px 24px;
    text-align   : center;
    margin       : 12px 0;
}
.churn-box   { background:rgba(232,89,60,0.12); border:2px solid #E8593C; }
.nochurn-box { background:rgba(29,158,117,0.12); border:2px solid #1D9E75; }

.badge {
    display      : inline-block;
    padding      : 4px 14px;
    border-radius: 20px;
    font-size    : 0.85rem;
    font-weight  : 600;
    margin-top   : 10px;
}
.sec-label {
    font-size    : 0.78rem;
    font-weight  : 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity      : 0.5;
    margin-bottom: 6px;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-size : 1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải model AI...")
def load_model():
    try:
        model, scaler = load_model_bundle('best_churn_model.pkl')
        return model, scaler, None
    except FileNotFoundError:
        return None, None, "Không tìm thấy `best_churn_model.pkl`."
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_model()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Banking Churn AI")
    st.caption("Dự đoán khách hàng rời bỏ ngân hàng")
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
            "3. Đặt cùng thư mục với `app.py`"
        )

    st.divider()
    st.markdown("""
**3 tính năng chính:**
- 🔮 **Đơn lẻ** — Nhập thông tin 1 KH → dự đoán + SHAP
- 📤 **Batch** — Upload CSV → dự đoán hàng loạt
- 📊 **Dashboard** — Xem hiệu suất mô hình
    """)
    st.divider()
    st.caption("📘 Project cuối kỳ AI  \nDataset: Banking Churn Prediction  \nKaggle: trnhuytun")


# ─────────────────────────────────────────────────────────────
# 3 TABS
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

    st.markdown("### 🔮 Nhập thông tin khách hàng ngân hàng")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        # Cột 1 — Nhân khẩu học
        with col1:
            st.markdown('<p class="sec-label">👤 Nhân khẩu học</p>', unsafe_allow_html=True)
            geography  = st.selectbox("Quốc gia", ["France", "Germany", "Spain"],
                                      help="Khách hàng đang sinh sống tại")
            gender     = st.selectbox("Giới tính", ["Male", "Female"])
            age        = st.slider("Tuổi", 18, 92, 38)
            credit_score = st.slider("Điểm tín dụng", 300, 850, 600,
                                     help="Credit score: 300 (rất thấp) → 850 (xuất sắc)")

        # Cột 2 — Thông tin tài khoản
        with col2:
            st.markdown('<p class="sec-label">🏦 Thông tin tài khoản</p>', unsafe_allow_html=True)
            tenure      = st.slider("Số năm gắn bó với ngân hàng", 0, 10, 3)
            balance     = st.number_input("Số dư tài khoản ($)", 0.0, 300000.0, 75000.0,
                                          step=500.0,
                                          help="Số dư hiện tại trong tài khoản")
            num_products = st.selectbox("Số sản phẩm đang dùng", [1, 2, 3, 4],
                                        help="VD: tài khoản tiết kiệm, thẻ tín dụng, vay...")
            estimated_salary = st.number_input("Lương ước tính ($)", 0.0, 300000.0, 80000.0,
                                               step=1000.0)

        # Cột 3 — Trạng thái
        with col3:
            st.markdown('<p class="sec-label">📋 Trạng thái thành viên</p>', unsafe_allow_html=True)
            has_cr_card = st.selectbox("Có thẻ tín dụng?", [1, 0],
                                       format_func=lambda x: "Có" if x else "Không")
            is_active   = st.selectbox("Thành viên tích cực?", [1, 0],
                                       format_func=lambda x: "Có" if x else "Không",
                                       help="Thường xuyên sử dụng dịch vụ trong 6 tháng qua")

            # Thông tin tự tính
            st.divider()
            st.markdown("**📊 Features tự tính:**")
            bal_per_prod = balance / (num_products + 1)
            zero_bal     = 1 if balance == 0 else 0
            st.metric("Số dư / Sản phẩm", f"${bal_per_prod:,.0f}")
            st.metric("Số dư bằng 0", "Có" if zero_bal else "Không")

        submitted = st.form_submit_button(
            "🚀  Dự đoán ngay",
            use_container_width=True,
            type="primary",
        )

    # ── KẾT QUẢ ─────────────────────────────────────────────
    if submitted:
        customer_raw = {
            'CreditScore':     credit_score,
            'Geography':       geography,
            'Gender':          gender,
            'Age':             age,
            'Tenure':          tenure,
            'Balance':         balance,
            'NumOfProducts':   num_products,
            'HasCrCard':       has_cr_card,
            'IsActiveMember':  is_active,
            'EstimatedSalary': estimated_salary,
        }

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
                <div style="opacity:.65;font-size:0.9rem;margin-top:4px">xác suất rời bỏ ngân hàng</div>
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
                    "ưu đãi giữ chân: lãi suất tốt hơn, miễn phí dịch vụ, "
                    "tặng điểm thưởng hoặc nâng cấp gói."
                )
            elif prob >= 0.4:
                st.warning(
                    "⚡ **Theo dõi sát** — Gửi khảo sát hài lòng, giới thiệu "
                    "thêm sản phẩm phù hợp (vay mua nhà, bảo hiểm)."
                )
            else:
                st.success(
                    "✅ **Khách hàng ổn định** — Duy trì chất lượng dịch vụ, "
                    "có thể cross-sell thêm sản phẩm tài chính."
                )

            with st.expander("📋 Thông tin đã nhập"):
                summary = {
                    "Quốc gia": geography, "Giới tính": gender, "Tuổi": age,
                    "Credit Score": credit_score, "Tenure": f"{tenure} năm",
                    "Số dư": f"${balance:,.0f}", "Số SP": num_products,
                    "Lương": f"${estimated_salary:,.0f}",
                }
                for k, v in summary.items():
                    st.text(f"  {k}: {v}")

        with right:
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                number= {'suffix': '%', 'font': {'size': 40}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar' : {'color': risk_color, 'thickness': 0.22},
                    'steps': [
                        {'range': [0,  30], 'color': 'rgba(29,158,117,.15)'},
                        {'range': [30, 60], 'color': 'rgba(243,156,18,.15)'},
                        {'range': [60,100], 'color': 'rgba(232,89,60,.15)'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 3},
                        'thickness': 0.82, 'value': 50,
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

            q1, q2 = st.columns(2)
            q1.metric("Credit Score", credit_score)
            q2.metric("Tenure", f"{tenure} năm")
            q1.metric("Số dư", f"${balance:,.0f}")
            q2.metric("Số SP", num_products)

        # ── SHAP ──────────────────────────────────────────
        st.divider()
        st.markdown("#### 🔍 Giải thích dự đoán — SHAP Waterfall")
        st.caption(
            "Thanh đỏ = feature đẩy xác suất churn **lên** | "
            "Thanh xanh = feature kéo xác suất **xuống**"
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
                max_display=12, show=False,
            )
            plt.tight_layout()
            st.pyplot(fig_w, use_container_width=True)
            plt.close()

        except Exception:
            st.info("Hiển thị Feature Importance thay thế cho SHAP.")
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                fi_df = pd.DataFrame({
                    'Feature':    get_vi_feature_names(FEATURE_ORDER),
                    'Importance': imp,
                }).sort_values('Importance').tail(12)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='Blues')
                fig_fi.update_layout(
                    height=380, showlegend=False,
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
    st.caption("Upload CSV cùng cấu trúc Banking Churn dataset — hệ thống tự predict toàn bộ.")

    # Template
    sample_row = {
        'RowNumber': 1, 'CustomerId': 15634602, 'Surname': 'Hargrave',
        'CreditScore': 619, 'Geography': 'France', 'Gender': 'Female',
        'Age': 42, 'Tenure': 2, 'Balance': 0.0, 'NumOfProducts': 1,
        'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 101348.88,
    }
    template_df = pd.DataFrame([sample_row])

    _, br = st.columns([3, 1])
    with br:
        st.download_button(
            "⬇️ Tải file CSV mẫu",
            data      = template_df.to_csv(index=False).encode('utf-8'),
            file_name = "template_bank_churn.csv",
            mime      = "text/csv",
        )

    uploaded = st.file_uploader("📁 Upload file CSV khách hàng", type=["csv"])

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.success(f"✅ Đã tải **{len(df_up):,}** khách hàng")

        with st.expander("👁️ Xem trước 5 dòng đầu"):
            st.dataframe(df_up.head(), use_container_width=True)

        if st.button("🚀 Chạy dự đoán batch", type="primary"):
            with st.spinner(f"⏳ Đang xử lý {len(df_up):,} khách hàng..."):
                result_df = predict_batch(model, scaler, df_up)

            n_total    = len(result_df)
            n_churn    = (result_df['Prediction'] == 'CHURN').sum()
            churn_rate = n_churn / n_total
            high_risk  = (result_df['Risk_Level'] == '🔴 Cao').sum()

            m1, m2, m3, m4 = st.columns(4)
            for col, (lbl, val, color) in zip(
                [m1, m2, m3, m4],
                [
                    ("Tổng khách hàng", f"{n_total:,}",    "#3266ad"),
                    ("Dự đoán CHURN",   f"{n_churn:,}",    "#E8593C"),
                    ("Tỉ lệ churn",     f"{churn_rate:.1%}","#f39c12"),
                    ("Nguy cơ cao",     f"{high_risk:,}",  "#E8593C"),
                ]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="lbl">{lbl}</div>
                        <div class="val" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("")
            ch1, ch2 = st.columns(2)

            with ch1:
                risk_cnt = result_df['Risk_Level'].value_counts()
                fig_pie  = go.Figure(go.Pie(
                    labels        = risk_cnt.index.tolist(),
                    values        = risk_cnt.values.tolist(),
                    hole          = 0.44,
                    marker_colors = ['#1D9E75', '#f39c12', '#E8593C'],
                    textinfo      = 'label+percent',
                ))
                fig_pie.update_layout(
                    title="Phân bố nguy cơ", height=320,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white', showlegend=False,
                    margin=dict(l=10,r=10,t=50,b=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch2:
                fig_hist = px.histogram(
                    result_df, x='Churn_Probability', nbins=30,
                    color='Prediction',
                    color_discrete_map={'CHURN':'#E8593C','KHÔNG CHURN':'#1D9E75'},
                    title="Phân phối xác suất Churn",
                    labels={'Churn_Probability':'Xác suất','count':'Số KH'},
                )
                fig_hist.update_layout(
                    height=320, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', legend_title="",
                    margin=dict(l=10,r=10,t=50,b=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Scatter: Age vs Balance theo nguy cơ
            if 'Age' in result_df.columns and 'Balance' in result_df.columns:
                fig_sc = px.scatter(
                    result_df.head(500),
                    x='Age', y='Balance',
                    color='Prediction',
                    color_discrete_map={'CHURN':'#E8593C','KHÔNG CHURN':'#1D9E75'},
                    size='Churn_Probability',
                    hover_data=['CreditScore','NumOfProducts'] if 'CreditScore' in result_df.columns else None,
                    title="Phân bố Tuổi vs Số dư (top 500 KH)",
                )
                fig_sc.update_layout(
                    height=340, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            # Bảng kết quả
            st.markdown("#### 📋 Kết quả chi tiết (top 50 nguy cơ cao nhất)")
            id_col = 'CustomerId' if 'CustomerId' in result_df.columns else None
            base   = ['Churn_Probability','Prediction','Risk_Level',
                      'Age','CreditScore','Balance','NumOfProducts','Geography']
            show   = ([id_col]+base) if id_col else base
            show   = [c for c in show if c in result_df.columns]

            st.dataframe(
                result_df[show].head(50)
                    .style
                    .background_gradient(subset=['Churn_Probability'], cmap='RdYlGn_r', vmin=0, vmax=1)
                    .format({'Churn_Probability':'{:.1%}', 'Balance':'${:,.0f}'}),
                use_container_width=True,
                height=400,
            )

            st.download_button(
                "⬇️ Tải kết quả đầy đủ (CSV)",
                data      = result_df.to_csv(index=False).encode('utf-8'),
                file_name = "bank_churn_predictions.csv",
                mime      = "text/csv",
            )


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Thông tin & Hiệu suất Mô hình")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("#### 🏗️ Pipeline xử lý")
        pipeline_data = {
            "Bước": [
                "1. Load data",
                "2. Drop cột thừa",
                "3. Encode categorical",
                "4. Feature engineering",
                "5. Scale features",
                "6. Cân bằng class",
                "7. Train models",
                "8. Ensemble",
                "9. Explainability",
            ],
            "Kỹ thuật": [
                "Banking Churn CSV → DataFrame",
                "RowNumber, CustomerId, Surname",
                "LabelEncoder (Geography, Gender)",
                "BalancePerProduct, ZeroBalance",
                "StandardScaler",
                "SMOTE (oversampling)",
                "LR · DT · RF · XGB · SVM",
                "Soft VotingClassifier",
                "SHAP TreeExplainer",
            ],
        }
        st.dataframe(pd.DataFrame(pipeline_data), use_container_width=True, hide_index=True)

    with d2:
        st.markdown("#### 📐 Thông số Dataset")
        dataset_data = {
            "Thông số": [
                "Dataset", "Nguồn", "Tổng mẫu",
                "Features gốc", "Features sau engineer", "Tỉ lệ Churn",
                "Train / Test", "Cross-validation",
            ],
            "Giá trị": [
                "Banking Customer Churn",
                "Kaggle — trnhuytun",
                "10.000 khách hàng",
                "14 cột (10 features)",
                "12 features (+BalancePerProduct, +ZeroBalance)",
                "~20% (mất cân bằng)",
                "80% / 20% (stratified)",
                "StratifiedKFold (k=5)",
            ],
        }
        st.dataframe(pd.DataFrame(dataset_data), use_container_width=True, hide_index=True)

    st.divider()

    # Performance chart
    st.markdown("#### 🎯 So sánh AUC-ROC các mô hình")
    perf = {
        'Model':    ['Logistic Regression','Decision Tree','Random Forest',
                     'XGBoost','SVM','🏆 Soft Voting'],
        'AUC-ROC':  [0.771, 0.726, 0.855, 0.872, 0.791, 0.876],
        'F1-Score': [0.487, 0.521, 0.606, 0.625, 0.511, 0.631],
        'Recall':   [0.421, 0.503, 0.558, 0.581, 0.447, 0.588],
        'Precision':[0.578, 0.541, 0.664, 0.678, 0.598, 0.681],
    }
    perf_df = pd.DataFrame(perf)

    chart_metric = st.selectbox("Chọn metric:", ['AUC-ROC','F1-Score','Recall','Precision'])
    colors = ['#ffd700' if '🏆' in m else '#3266ad' for m in perf_df['Model']]

    fig_bar = go.Figure(go.Bar(
        x=perf_df['Model'], y=perf_df[chart_metric],
        marker_color=colors,
        text=perf_df[chart_metric].round(3),
        textposition='outside',
    ))
    fig_bar.update_layout(
        yaxis_range=[perf_df[chart_metric].min()-0.05, perf_df[chart_metric].max()+0.06],
        height=380, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font_color='white',
        xaxis_tickangle=-20, showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 📋 Bảng kết quả đầy đủ")
    st.dataframe(
        perf_df.set_index('Model').style
            .highlight_max(axis=0, color='rgba(29,158,117,0.3)')
            .format('{:.4f}'),
        use_container_width=True,
    )

    st.divider()

    # Feature importance — banking context
    st.markdown("#### 🔍 Feature Importance (XGBoost — ước tính)")
    fi_data = {
        'Feature':    ['Tuổi','Số dư / Sản phẩm','Số dư tài khoản',
                       'Điểm tín dụng','Số dư bằng 0','Số năm gắn bó',
                       'Số SP đang dùng','Lương ước tính','Quốc gia',
                       'Thành viên tích cực','Giới tính','Có thẻ tín dụng'],
        'Importance': [0.241, 0.187, 0.142, 0.098, 0.087, 0.071,
                       0.063, 0.044, 0.032, 0.021, 0.009, 0.005],
    }
    fi_df = pd.DataFrame(fi_data).sort_values('Importance')
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues',
                    title="Top 12 Feature Importance")
    fig_fi.update_layout(
        height=420, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info(
        "**Insight chính từ dataset Banking:**  \n"
        "- **Tuổi** là yếu tố dự đoán mạnh nhất — KH lớn tuổi có xu hướng churn cao hơn  \n"
        "- **Số dư = 0** là tín hiệu nguy hiểm — KH không dùng tài khoản có nguy cơ rời bỏ  \n"
        "- **Quốc gia Germany** có churn rate cao nhất (~32%) so với France (~16%) và Spain (~17%)  \n"
        "- **Thành viên tích cực** giảm đáng kể xác suất churn"
    )