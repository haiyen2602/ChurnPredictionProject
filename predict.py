"""
predict.py — Backend inference module
Dataset : VIB Banking Churn (trnhuytun/churn-prediction-dataset)
Target  : Churn (0 = ở lại, 1 = rời bỏ)
Note    : Tất cả features ĐÃ là số trong dataset gốc → không cần encode
"""

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────
# FEATURE ORDER — phải khớp 100% với scaler.feature_names_in_
# Lấy từ: list(scaler.feature_names_in_)
# ─────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    'Client_gender',
    'Age', 
    'Tenure', 
    'SMS',
    'Type_Transactions',
    'Total_trans_no',
    'Avg_Trans_no_month', 
    'Avg_Trans_Amount', 
    'Avg_CurrentAccount_Balance', 
    'Avg_TermDeposit_Balance', 
    'Avg_Loan_Balance', 
    'Max_Loan_Balance', 
    'No_CC', 
    'No_DC'
]

# Nhãn tiếng Việt dùng trong SHAP plot
FEATURE_LABELS = {
    'Client_gender':             'Giới tính (0=Nữ, 1=Nam)',
    'Age':                       'Tuổi',
    'Staff_VIB':                 'Là nhân viên VIB',
    'Tenure':                    'Thâm niên gắn bó (năm)',
    'SMS':                       'Đăng ký SMS Banking',
    'Verify_method':             'Phương thức xác thực',
    'EB_register_channel':       'Kênh đăng ký EB',
    'No_Activity_Name':          'Số loại hoạt động',
    'Type_Transactions':         'Loại giao dịch',
    'Total_trans_no':            'Tổng số giao dịch',
    'Avg_Trans_no_month':        'TB giao dịch/tháng',
    'Avg_Trans_Amount':          'TB giá trị GD (VND)',
    'Max_Trans_Amount':          'GD lớn nhất (VND)',
    'Min_Trans_Amount':          'GD nhỏ nhất (VND)',
    'No_CurrentAccount':         'Số TK thanh toán',
    'Avg_CurrentAccount_Balance':'TB số dư TK TT (VND)',
    'Max_CurrentAccount_Balance':'Số dư TK TT cao nhất',
    'Min_CurrentAccount_Balance':'Số dư TK TT thấp nhất',
    'No_TermDeposit':            'Số TK tiết kiệm',
    'Avg_TermDeposit_Balance':   'TB số dư TK TK (VND)',
    'Max_TermDeposit_Balance':   'Số dư TK TK cao nhất',
    'Min_TermDeposit_Balance':   'Số dư TK TK thấp nhất',
    'No_Loan':                   'Số khoản vay',
    'Avg_Loan_Balance':          'TB dư nợ vay (VND)',
    'Max_Loan_Balance':          'Dư nợ vay cao nhất',
    'Min_Loan_Balance':          'Dư nợ vay thấp nhất',
    'No_CC':                     'Số thẻ tín dụng',
    'No_DC':                     'Số thẻ ghi nợ',
}

# Cột cần loại bỏ khi xử lý batch CSV
DROP_COLS = ['Customer_number', 'Churn']


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_model_bundle(path: str = 'best_churn_model.pkl'):
    """
    Load model + scaler từ file .pkl.
    Tạo trong notebook bằng:
        joblib.dump({'model': best_model, 'scaler': scaler}, 'best_churn_model.pkl')
    """
    bundle = joblib.load(path)
    return bundle['model'], bundle['scaler']


# ─────────────────────────────────────────────────────────────
# PREPROCESS — không cần encode, chỉ cần scale
# ─────────────────────────────────────────────────────────────

def preprocess_single(customer_dict: dict, scaler):
    """
    Nhận dict các giá trị số → tạo DataFrame đúng thứ tự → scale.
    Trả về (X_scaled: ndarray, X_df: DataFrame)

    Input:  {'Client_gender': 1, 'Age': 35, 'Tenure': 5.2, ...}
    Output: X_scaled (1×28), X_df (1×28 với tên cột)
    """
    X_df     = pd.DataFrame([customer_dict])[FEATURE_ORDER]
    X_scaled = scaler.transform(X_df)
    return X_scaled, X_df


def preprocess_batch(df_raw: pd.DataFrame, scaler):
    """
    Xử lý batch DataFrame từ file CSV.
    Trả về (X_scaled: ndarray, X_df: DataFrame)
    """
    df = df_raw.copy()

    # Bỏ cột định danh và target nếu có
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    X_df     = df[FEATURE_ORDER]
    X_scaled = scaler.transform(X_df)
    return X_scaled, X_df


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict_single(model, scaler, customer_dict: dict):
    """
    Dự đoán 1 khách hàng.
    Returns: (prob, label, X_scaled, X_df)
    """
    X_scaled, X_df = preprocess_single(customer_dict, scaler)
    prob  = model.predict_proba(X_scaled)[0][1]
    label = 'CHURN' if prob > 0.5 else 'KHÔNG CHURN'
    return prob, label, X_scaled, X_df


def predict_batch(model, scaler, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Dự đoán hàng loạt từ DataFrame CSV.
    Trả về DataFrame gốc + cột Churn_Probability, Prediction, Risk_Level.
    """
    X_scaled, _ = preprocess_batch(df_raw, scaler)
    probs  = model.predict_proba(X_scaled)[:, 1]
    labels = ['CHURN' if p > 0.5 else 'KHÔNG CHURN' for p in probs]

    result = df_raw.copy()
    result['Churn_Probability'] = np.round(probs, 4)
    result['Prediction']        = labels
    result['Risk_Level']        = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['🟢 Thấp', '🟡 Trung bình', '🔴 Cao'],
    )
    return result.sort_values('Churn_Probability', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple:
    """Trả về (emoji, label, màu hex) theo xác suất."""
    if prob >= 0.7:
        return '🔴', 'Nguy cơ CAO', '#E8593C'
    elif prob >= 0.4:
        return '🟡', 'Nguy cơ TRUNG BÌNH', '#f39c12'
    else:
        return '🟢', 'Nguy cơ THẤP', '#1D9E75'


def get_vi_feature_names(feature_list: list) -> list:
    """Chuyển tên feature sang tiếng Việt cho SHAP plot."""
    return [FEATURE_LABELS.get(f, f) for f in feature_list]


def fmt_vnd(value: float) -> str:
    """Format số tiền VND cho dễ đọc."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f} tỷ"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.0f} triệu"
    else:
        return f"{value:,.0f} đ"