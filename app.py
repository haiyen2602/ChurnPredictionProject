"""
app.py — Streamlit Web App | VIB Banking Churn Prediction
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
    fmt_vnd,
    FEATURE_ORDER,
    FEATURE_LABELS,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "VIB Churn Prediction AI",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background   : var(--background-color, #1a1f2e);
    border       : 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding      : 20px 24px;
    text-align   : center;
    margin-bottom: 8px;
}
.metric-card .val { font-size:2rem; font-weight:700; margin:6px 0; }
.metric-card .lbl { font-size:0.75rem; opacity:.55; text-transform:uppercase; letter-spacing:1px; }
.result-box  { border-radius:16px; padding:28px 24px; text-align:center; margin:12px 0; }
.churn-box   { background:rgba(232,89,60,0.12); border:2px solid #E8593C; }
.nochurn-box { background:rgba(29,158,117,0.12); border:2px solid #1D9E75; }
.badge { display:inline-block; padding:4px 14px; border-radius:20px;
         font-size:0.85rem; font-weight:600; margin-top:10px; }
.sec-label { font-size:0.78rem; font-weight:700; letter-spacing:1px;
             text-transform:uppercase; opacity:.5; margin-bottom:4px; }
div[data-testid="stTabs"] button[role="tab"] { font-size:1rem; font-weight:600; }
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
    st.markdown("## 🏦 VIB Churn AI")
    st.caption("Dự đoán khách hàng rời bỏ Ngân hàng VIB")
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
**3 tính năng:**
- 🔮 **Đơn lẻ** — Nhập thông tin 1 KH → dự đoán + SHAP
- 📤 **Batch** — Upload CSV → dự đoán hàng loạt
- 📊 **Dashboard** — Xem hiệu suất mô hình
    """)
    st.divider()
    st.caption("📘 Project cuối kỳ AI  \nDataset: VIB Churn Prediction  \nKaggle: trnhuytun")


# ─────────────────────────────────────────────────────────────
# TABS
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

    st.markdown("### 🔮 Nhập thông tin khách hàng VIB")
    st.caption("Điền đầy đủ 28 features → nhấn **Dự đoán ngay**")

    with st.form("predict_form"):
        # ── NHÓM 1: Thông tin khách hàng ─────────────────
        st.markdown('<p class="sec-label">👤 Thông tin khách hàng</p>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            client_gender = st.selectbox("Giới tính", [0, 1],
                                         format_func=lambda x: "Nam" if x else "Nữ")
        with r1c2:
            age = st.number_input("Tuổi", 18, 90, 38, step=1)
        with r1c3:
            staff_vib = st.selectbox("Là nhân viên VIB?", [0, 1],
                                     format_func=lambda x: "Có" if x else "Không")
        with r1c4:
            tenure = st.number_input("Thâm niên (năm)", 0.0, 50.0, 5.0, step=0.5)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            sms = st.selectbox("Đăng ký SMS Banking", [0, 1],
                               format_func=lambda x: "Có" if x else "Không")
        with r2c2:
            verify_method = st.selectbox("Phương thức xác thực",
                                         [0, 1, 2, 3],
                                         format_func=lambda x: f"Loại {x}")
        with r2c3:
            eb_channel = st.selectbox("Kênh đăng ký E-Banking",
                                      [0, 1, 2, 3, 4],
                                      format_func=lambda x: f"Kênh {x}")

        st.divider()

        # ── NHÓM 2: Hành vi giao dịch ────────────────────
        st.markdown('<p class="sec-label">💳 Hành vi giao dịch</p>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            no_activity   = st.number_input("Số loại hoạt động", 0, 50, 10, step=1)
        with t2:
            type_trans    = st.number_input("Loại giao dịch", 0, 10, 3, step=1)
        with t3:
            total_trans   = st.number_input("Tổng số GD", 0, 500, 20, step=1)
        with t4:
            avg_trans_month = st.number_input("TB GD/tháng", 0.0, 100.0, 1.5, step=0.1)

        t5, t6, t7 = st.columns(3)
        with t5:
            avg_trans_amt = st.number_input("TB giá trị GD (triệu VND)",
                                            0.0, 10000.0, 5.0, step=0.5,
                                            help="Nhập theo đơn vị triệu đồng")
        with t6:
            max_trans_amt = st.number_input("GD lớn nhất (triệu VND)",
                                            0.0, 100000.0, 10.0, step=1.0)
        with t7:
            min_trans_amt = st.number_input("GD nhỏ nhất (triệu VND)",
                                            0.0, 10000.0, 0.1, step=0.1)

        st.divider()

        # ── NHÓM 3: Tài khoản & Tiết kiệm ───────────────
        st.markdown('<p class="sec-label">🏦 Tài khoản thanh toán & Tiết kiệm</p>', unsafe_allow_html=True)
        ca1, ca2, ca3, ca4 = st.columns(4)
        with ca1:
            no_ca    = st.number_input("Số TK thanh toán", 0, 10, 1, step=1)
        with ca2:
            avg_ca   = st.number_input("TB số dư TK TT (triệu)", 0.0, 100000.0, 20.0, step=1.0)
        with ca3:
            max_ca   = st.number_input("Số dư TK TT cao nhất (triệu)", 0.0, 100000.0, 30.0, step=1.0)
        with ca4:
            min_ca   = st.number_input("Số dư TK TT thấp nhất (triệu)", 0.0, 100000.0, 5.0, step=0.5)

        td1, td2, td3, td4 = st.columns(4)
        with td1:
            no_td    = st.number_input("Số TK tiết kiệm", 0, 10, 0, step=1)
        with td2:
            avg_td   = st.number_input("TB số dư TK TK (triệu)", 0.0, 1000000.0, 0.0, step=1.0)
        with td3:
            max_td   = st.number_input("Số dư TK TK cao nhất (triệu)", 0.0, 1000000.0, 0.0, step=1.0)
        with td4:
            min_td   = st.number_input("Số dư TK TK thấp nhất (triệu)", 0.0, 1000000.0, 0.0, step=1.0)

        st.divider()

        # ── NHÓM 4: Khoản vay & Thẻ ─────────────────────
        st.markdown('<p class="sec-label">📋 Khoản vay & Thẻ</p>', unsafe_allow_html=True)
        ln1, ln2, ln3, ln4 = st.columns(4)
        with ln1:
            no_loan  = st.number_input("Số khoản vay", 0, 10, 0, step=1)
        with ln2:
            avg_loan = st.number_input("TB dư nợ vay (triệu)", 0.0, 1000000.0, 0.0, step=1.0)
        with ln3:
            max_loan = st.number_input("Dư nợ vay cao nhất (triệu)", 0.0, 1000000.0, 0.0, step=1.0)
        with ln4:
            min_loan = st.number_input("Dư nợ vay thấp nhất (triệu)", 0.0, 1000000.0, 0.0, step=1.0)

        card1, card2, _ = st.columns(3)
        with card1:
            no_cc = st.number_input("Số thẻ tín dụng (Credit)", 0, 10, 1, step=1)
        with card2:
            no_dc = st.number_input("Số thẻ ghi nợ (Debit)", 0, 10, 1, step=1)

        st.markdown("")
        submitted = st.form_submit_button("🚀  Dự đoán ngay",
                                          use_container_width=True,
                                          type="primary")

    # ── KẾT QUẢ ─────────────────────────────────────────────
    if submitted:
        # Nhân với 1_000_000 vì form nhập theo triệu VND
        M = 1_000_000
        customer_dict = {
            'Client_gender':             float(client_gender),
            'Age':                       float(age),
            'Staff_VIB':                 float(staff_vib),
            'Tenure':                    float(tenure),
            'SMS':                       float(sms),
            'Verify_method':             float(verify_method),
            'EB_register_channel':       float(eb_channel),
            'No_Activity_Name':          float(no_activity),
            'Type_Transactions':         float(type_trans),
            'Total_trans_no':            float(total_trans),
            'Avg_Trans_no_month':        float(avg_trans_month),
            'Avg_Trans_Amount':          avg_trans_amt * M,
            'Max_Trans_Amount':          max_trans_amt * M,
            'Min_Trans_Amount':          min_trans_amt * M,
            'No_CurrentAccount':         float(no_ca),
            'Avg_CurrentAccount_Balance': avg_ca * M,
            'Max_CurrentAccount_Balance': max_ca * M,
            'Min_CurrentAccount_Balance': min_ca * M,
            'No_TermDeposit':            float(no_td),
            'Avg_TermDeposit_Balance':   avg_td * M,
            'Max_TermDeposit_Balance':   max_td * M,
            'Min_TermDeposit_Balance':   min_td * M,
            'No_Loan':                   float(no_loan),
            'Avg_Loan_Balance':          avg_loan * M,
            'Max_Loan_Balance':          max_loan * M,
            'Min_Loan_Balance':          min_loan * M,
            'No_CC':                     float(no_cc),
            'No_DC':                     float(no_dc),
        }

        prob, label, X_scaled, X_df = predict_single(model, scaler, customer_dict)
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
                <div style="opacity:.65;font-size:.9rem;margin-top:4px">xác suất rời bỏ VIB</div>
                <span class="badge" style="background:{risk_color}22;
                      color:{risk_color};border:1px solid {risk_color}66">
                    {risk_label}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**💡 Khuyến nghị hành động:**")
            if prob >= 0.7:
                st.error(
                    "🚨 **Ưu tiên CAN THIỆP NGAY** — Gọi điện chăm sóc, "
                    "đề xuất ưu đãi lãi suất, miễn phí giao dịch, "
                    "hoặc tặng điểm thưởng để giữ chân khách hàng."
                )
            elif prob >= 0.4:
                st.warning(
                    "⚡ **Theo dõi & chủ động tiếp cận** — Gửi thông báo ưu đãi, "
                    "nhắc nhở sử dụng dịch vụ, giới thiệu sản phẩm mới phù hợp."
                )
            else:
                st.success(
                    "✅ **Khách hàng ổn định** — Duy trì trải nghiệm tốt, "
                    "có thể cross-sell thêm sản phẩm tiết kiệm/đầu tư."
                )

            with st.expander("📋 Xem lại dữ liệu đã nhập"):
                summary_df = pd.DataFrame([{
                    'Feature': FEATURE_LABELS.get(k, k),
                    'Giá trị': (fmt_vnd(v) if k in [
                        'Avg_Trans_Amount','Max_Trans_Amount','Min_Trans_Amount',
                        'Avg_CurrentAccount_Balance','Max_CurrentAccount_Balance',
                        'Min_CurrentAccount_Balance','Avg_TermDeposit_Balance',
                        'Max_TermDeposit_Balance','Min_TermDeposit_Balance',
                        'Avg_Loan_Balance','Max_Loan_Balance','Min_Loan_Balance'
                    ] else str(v))
                } for k, v in customer_dict.items()])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

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
            q1.metric("Số GD", int(total_trans))
            q2.metric("Tenure", f"{tenure:.1f} năm")
            q1.metric("TB GD/tháng", f"{avg_trans_month:.1f}")
            q2.metric("Thẻ CC/DC", f"{no_cc}/{no_dc}")

        # ── SHAP WATERFALL ────────────────────────────────
        st.divider()
        st.markdown("#### 🔍 Giải thích dự đoán — SHAP Waterfall")
        st.caption(
            "🔴 Thanh đỏ = feature đẩy xác suất churn **lên** | "
            "🔵 Thanh xanh = feature kéo xác suất **xuống**"
        )

        try:
            feat_names_vi = get_vi_feature_names(list(X_df.columns))
            explainer     = shap.TreeExplainer(model)
            shap_vals     = explainer.shap_values(X_df)

            fig_w, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            shap.waterfall_plot(
                shap.Explanation(
                    values        = shap_vals[0],
                    base_values   = explainer.expected_value,
                    data          = X_df.iloc[0].values,
                    feature_names = feat_names_vi,
                ),
                max_display=14, show=False,
            )
            plt.tight_layout()
            st.pyplot(fig_w, use_container_width=True)
            plt.close()

        except Exception as e:
            st.info(f"SHAP không khả dụng ({e}). Hiển thị Feature Importance.")
            if hasattr(model, 'feature_importances_'):
                fi_df = pd.DataFrame({
                    'Feature':    get_vi_feature_names(FEATURE_ORDER),
                    'Importance': model.feature_importances_,
                }).sort_values('Importance').tail(14)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='Blues')
                fig_fi.update_layout(height=420, showlegend=False,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        st.warning("⚠️ Cần file `best_churn_model.pkl`. Xem hướng dẫn ở sidebar.")
        st.stop()

    st.markdown("### 📤 Dự đoán hàng loạt từ file CSV")
    st.caption("Upload CSV cùng cấu trúc VIB dataset (28 cột features) — hệ thống tự predict toàn bộ.")

    # Template CSV mẫu
    template_row = {
        'Client_gender': 1, 'Age': 39.9, 'Staff_VIB': 0, 'Tenure': 6.8,
        'SMS': 0, 'Verify_method': 0, 'EB_register_channel': 0,
        'No_Activity_Name': 11, 'Type_Transactions': 3, 'Total_trans_no': 20,
        'Avg_Trans_no_month': 1.5, 'Avg_Trans_Amount': 5000000,
        'Max_Trans_Amount': 10000000, 'Min_Trans_Amount': 100000,
        'No_CurrentAccount': 1, 'Avg_CurrentAccount_Balance': 20000000,
        'Max_CurrentAccount_Balance': 30000000, 'Min_CurrentAccount_Balance': 5000000,
        'No_TermDeposit': 0, 'Avg_TermDeposit_Balance': 0,
        'Max_TermDeposit_Balance': 0, 'Min_TermDeposit_Balance': 0,
        'No_Loan': 0, 'Avg_Loan_Balance': 0,
        'Max_Loan_Balance': 0, 'Min_Loan_Balance': 0,
        'No_CC': 1, 'No_DC': 1,
    }
    template_df = pd.DataFrame([template_row])

    _, br = st.columns([3, 1])
    with br:
        st.download_button(
            "⬇️ Tải CSV mẫu",
            data      = template_df.to_csv(index=False).encode('utf-8'),
            file_name = "template_vib_churn.csv",
            mime      = "text/csv",
        )

    uploaded = st.file_uploader("📁 Upload file CSV", type=["csv"])

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
            for col, (lbl, val, color) in zip([m1, m2, m3, m4], [
                ("Tổng khách hàng", f"{n_total:,}",    "#3266ad"),
                ("Dự đoán CHURN",   f"{n_churn:,}",    "#E8593C"),
                ("Tỉ lệ Churn",     f"{churn_rate:.1%}","#f39c12"),
                ("Nguy cơ Cao",     f"{high_risk:,}",  "#E8593C"),
            ]):
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="lbl">{lbl}</div>
                        <div class="val" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("")
            ch1, ch2 = st.columns(2)

            with ch1:
                risk_cnt = result_df['Risk_Level'].value_counts()
                fig_pie  = go.Figure(go.Pie(
                    labels=risk_cnt.index.tolist(), values=risk_cnt.values.tolist(),
                    hole=0.44, marker_colors=['#1D9E75','#f39c12','#E8593C'],
                    textinfo='label+percent',
                ))
                fig_pie.update_layout(title="Phân bố nguy cơ", height=320,
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      font_color='white', showlegend=False,
                                      margin=dict(l=10,r=10,t=50,b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with ch2:
                fig_hist = px.histogram(
                    result_df, x='Churn_Probability', nbins=30, color='Prediction',
                    color_discrete_map={'CHURN':'#E8593C','KHÔNG CHURN':'#1D9E75'},
                    title="Phân phối xác suất Churn",
                )
                fig_hist.update_layout(
                    height=320, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font_color='white', legend_title="",
                    margin=dict(l=10,r=10,t=50,b=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Scatter Age vs Avg_Trans_Amount
            if 'Age' in result_df.columns:
                fig_sc = px.scatter(
                    result_df.head(500), x='Age', y='Avg_Trans_Amount',
                    color='Prediction',
                    color_discrete_map={'CHURN':'#E8593C','KHÔNG CHURN':'#1D9E75'},
                    size='Churn_Probability',
                    title="Phân bố Tuổi vs TB Giá trị GD (top 500 KH)",
                    labels={'Avg_Trans_Amount':'TB Giá trị GD (VND)'},
                )
                fig_sc.update_layout(
                    height=340, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            # Bảng kết quả
            st.markdown("#### 📋 Kết quả chi tiết (top 50 nguy cơ cao nhất)")
            id_col  = 'Customer_number' if 'Customer_number' in result_df.columns else None
            base    = ['Churn_Probability','Prediction','Risk_Level',
                       'Age','Tenure','Total_trans_no','Avg_Trans_Amount',
                       'No_CurrentAccount','No_CC','No_DC']
            show    = ([id_col]+base) if id_col else base
            show    = [c for c in show if c in result_df.columns]

            st.dataframe(
                result_df[show].head(50)
                    .style
                    .background_gradient(subset=['Churn_Probability'],
                                         cmap='RdYlGn_r', vmin=0, vmax=1)
                    .format({'Churn_Probability':'{:.1%}',
                             'Avg_Trans_Amount':'{:,.0f}'}),
                use_container_width=True, height=420,
            )

            st.download_button(
                "⬇️ Tải kết quả CSV",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name="vib_churn_predictions.csv",
                mime="text/csv",
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
            "Bước": ["1. Load data","2. Drop cột ID","3. Tách X & y",
                     "4. Train/Test split","5. SMOTE","6. StandardScaler",
                     "7. Train 5 models","8. Chọn best model","9. SHAP"],
            "Chi tiết": [
                "Dataset.csv → DataFrame",
                "Loại bỏ Customer_number",
                "X = tất cả features | y = Churn",
                "80/20 stratified",
                "Cân bằng class (chỉ trên train)",
                "Fit trên train, transform cả 2",
                "LR · DT · RF · XGB · SVM",
                "XGBoost (AUC cao nhất)",
                "TreeExplainer",
            ],
        }
        st.dataframe(pd.DataFrame(pipeline_data), use_container_width=True, hide_index=True)

    with d2:
        st.markdown("#### 📐 Thông số Dataset")
        dataset_data = {
            "Thông số": ["Dataset","Nguồn","Loại","Tổng features",
                         "Target","Train/Test","Xử lý imbalance"],
            "Giá trị":  ["VIB Banking Churn","Kaggle — trnhuytun",
                         "Banking (VN) · Synthetic","28 features (tất cả numeric)",
                         "Churn (0=ở lại, 1=rời bỏ)",
                         "80% / 20% (stratified)","SMOTE oversampling"],
        }
        st.dataframe(pd.DataFrame(dataset_data), use_container_width=True, hide_index=True)

    st.markdown("#### 📋 28 Features trong Dataset")
    feat_df = pd.DataFrame([
        {'Feature': k, 'Mô tả': v, 'Loại': 'Binary' if k in [
            'Client_gender','Staff_VIB','SMS'] else (
            'Category' if k in ['Verify_method','EB_register_channel',
                                 'No_Activity_Name','Type_Transactions'] else 'Numeric'
        )} for k, v in FEATURE_LABELS.items()
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=350)

    st.divider()
    st.markdown("#### 🎯 Hiệu suất các mô hình (ước tính)")
    perf = {
        'Model':    ['Logistic Regression','Decision Tree','Random Forest',
                     'XGBoost','SVM'],
        'AUC-ROC':  [0.784, 0.741, 0.851, 0.873, 0.802],
        'F1-Score': [0.512, 0.498, 0.601, 0.634, 0.531],
        'Recall':   [0.468, 0.481, 0.562, 0.598, 0.489],
        'Precision':[0.564, 0.516, 0.646, 0.674, 0.582],
    }
    perf_df = pd.DataFrame(perf)

    chart_metric = st.selectbox("Chọn metric:", ['AUC-ROC','F1-Score','Recall','Precision'])
    colors = ['#ffd700' if m == 'XGBoost' else '#3266ad' for m in perf_df['Model']]

    fig_bar = go.Figure(go.Bar(
        x=perf_df['Model'], y=perf_df[chart_metric],
        marker_color=colors,
        text=perf_df[chart_metric].round(3), textposition='outside',
    ))
    fig_bar.update_layout(
        yaxis_range=[perf_df[chart_metric].min()-0.05, perf_df[chart_metric].max()+0.06],
        height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', xaxis_tickangle=0, showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        perf_df.set_index('Model').style
            .highlight_max(axis=0, color='rgba(29,158,117,0.3)')
            .format('{:.4f}'),
        use_container_width=True,
    )

    st.info(
        "**Key insights từ VIB Dataset:**  \n"
        "- **Avg_Trans_Amount** & **Total_trans_no** — khách hàng giao dịch ít/thưa → nguy cơ churn cao  \n"
        "- **Tenure** thấp — mới dùng dịch vụ, chưa gắn bó → dễ rời bỏ  \n"
        "- **No_CC / No_DC = 0** — không có thẻ → ít liên kết với ngân hàng  \n"
        "- **SMS = 0** — chưa đăng ký thông báo → tương tác kém"
    )