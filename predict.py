"""
predict.py — Backend inference module
Dataset: Banking Customer Churn (trnhuytun/churn-prediction-dataset)
Target : Exited (0 = ở lại, 1 = rời bỏ)
"""

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────
# ENCODING MAPS — LabelEncoder alphabetical sort
# Banking dataset chỉ có 2 cột categorical: Geography, Gender
# ─────────────────────────────────────────────────────────────
ENCODING_MAP = {
    'Geography': {'France': 0, 'Germany': 1, 'Spain': 2},
    'Gender':    {'Female': 0, 'Male': 1},
}

# Thứ tự features phải khớp 100% với lúc train trong notebook
FEATURE_ORDER = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary',
    'BalancePerProduct',   # feature engineering
    'ZeroBalance',         # feature engineering
]

# Nhãn tiếng Việt cho SHAP plot
FEATURE_LABELS = {
    'CreditScore':      'Điểm tín dụng',
    'Geography':        'Quốc gia',
    'Gender':           'Giới tính',
    'Age':              'Tuổi',
    'Tenure':           'Số năm gắn bó',
    'Balance':          'Số dư tài khoản ($)',
    'NumOfProducts':    'Số sản phẩm đang dùng',
    'HasCrCard':        'Có thẻ tín dụng',
    'IsActiveMember':   'Thành viên tích cực',
    'EstimatedSalary':  'Lương ước tính ($)',
    'BalancePerProduct':'Số dư / Sản phẩm',
    'ZeroBalance':      'Số dư bằng 0',
}


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
# ENCODE & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def encode_single(customer_raw: dict) -> dict:
    """
    Encode 1 khách hàng từ raw values → số.
    Input:  {'Geography': 'France', 'Gender': 'Male', 'Age': 35, ...}
    Output: {'Geography': 0, 'Gender': 1, 'Age': 35, ...,
             'BalancePerProduct': ..., 'ZeroBalance': ...}
    """
    encoded = {}
    for key, val in customer_raw.items():
        if key in ENCODING_MAP:
            encoded[key] = ENCODING_MAP[key].get(str(val), 0)
        else:
            encoded[key] = val

    # Feature engineering — phải giống hệt notebook
    balance     = encoded.get('Balance', 0)
    n_products  = encoded.get('NumOfProducts', 1)
    encoded['BalancePerProduct'] = balance / (n_products + 1)
    encoded['ZeroBalance']       = 1 if balance == 0 else 0

    return encoded


def preprocess_single(customer_raw: dict, scaler):
    """Encode + scale 1 khách hàng → (X_scaled, X_df)."""
    encoded  = encode_single(customer_raw)
    X_df     = pd.DataFrame([encoded])[FEATURE_ORDER]
    X_scaled = scaler.transform(X_df)
    return X_scaled, X_df


def preprocess_batch(df_raw: pd.DataFrame, scaler):
    """
    Xử lý DataFrame nhiều khách hàng (cùng cấu trúc Banking CSV).
    Trả về (X_scaled, X_df)
    """
    df = df_raw.copy()

    # Bỏ cột không dùng
    for col in ['RowNumber', 'CustomerId', 'Surname', 'Exited']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical
    for col, mapping in ENCODING_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Feature engineering
    df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)
    df['ZeroBalance']       = (df['Balance'] == 0).astype(int)

    X_df     = df[FEATURE_ORDER]
    X_scaled = scaler.transform(X_df)
    return X_scaled, X_df


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict_single(model, scaler, customer_raw: dict):
    """
    Dự đoán 1 khách hàng.
    Returns: (prob, label, X_scaled, X_df)
    """
    X_scaled, X_df = preprocess_single(customer_raw, scaler)
    prob  = model.predict_proba(X_scaled)[0][1]
    label = 'CHURN' if prob > 0.5 else 'KHÔNG CHURN'
    return prob, label, X_scaled, X_df


def predict_batch(model, scaler, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Dự đoán hàng loạt. Trả về DataFrame + cột Churn_Probability,
    Prediction, Risk_Level, sắp xếp theo nguy cơ giảm dần.
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