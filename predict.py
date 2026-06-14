"""
predict.py — Backend inference module
Xử lý toàn bộ logic encode, preprocess, predict
Không phụ thuộc notebook — chỉ cần file best_churn_model.pkl
"""

import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────
# ENCODING MAPS
# LabelEncoder sort theo alphabet — phải khớp chính xác với notebook
# ─────────────────────────────────────────────────────────────
ENCODING_MAP = {
    'gender':           {'Female': 0, 'Male': 1},
    'Partner':          {'No': 0, 'Yes': 1},
    'Dependents':       {'No': 0, 'Yes': 1},
    'PhoneService':     {'No': 0, 'Yes': 1},
    'MultipleLines':    {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService':  {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity':   {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup':     {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport':      {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV':      {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies':  {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {
        'Month-to-month': 0,
        'One year':        1,
        'Two year':        2,
    },
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)':   1,
        'Electronic check':          2,
        'Mailed check':              3,
    },
}

# Thứ tự features phải khớp 100% với lúc train trong notebook
FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
    'ChargePerTenure', 'NumServices',
]

# Các cột dịch vụ dùng để tính NumServices
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

# Nhãn hiển thị tiếng Việt cho từng feature (dùng trong SHAP plot)
FEATURE_LABELS = {
    'gender':           'Giới tính',
    'SeniorCitizen':    'Người cao tuổi',
    'Partner':          'Có đối tác',
    'Dependents':       'Có người phụ thuộc',
    'tenure':           'Số tháng sử dụng',
    'PhoneService':     'Dịch vụ điện thoại',
    'MultipleLines':    'Nhiều đường dây',
    'InternetService':  'Dịch vụ Internet',
    'OnlineSecurity':   'Bảo mật Online',
    'OnlineBackup':     'Sao lưu Online',
    'DeviceProtection': 'Bảo vệ thiết bị',
    'TechSupport':      'Hỗ trợ kỹ thuật',
    'StreamingTV':      'Xem TV trực tuyến',
    'StreamingMovies':  'Xem phim trực tuyến',
    'Contract':         'Loại hợp đồng',
    'PaperlessBilling': 'Hóa đơn điện tử',
    'PaymentMethod':    'Phương thức thanh toán',
    'MonthlyCharges':   'Chi phí hàng tháng ($)',
    'TotalCharges':     'Tổng chi phí ($)',
    'ChargePerTenure':  'Chi phí / Tháng sử dụng',
    'NumServices':      'Số dịch vụ đang dùng',
}


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

def load_model_bundle(path: str = 'best_churn_model.pkl'):
    """
    Load model + scaler từ file .pkl.
    File được tạo trong notebook bằng:
        joblib.dump({'model': best_model, 'scaler': scaler}, 'best_churn_model.pkl')
    """
    bundle = joblib.load(path)
    return bundle['model'], bundle['scaler']


# ─────────────────────────────────────────────────────────────
# ENCODE & PREPROCESS
# ─────────────────────────────────────────────────────────────

def encode_single(customer_raw: dict) -> dict:
    """
    Encode 1 khách hàng từ raw string labels → số nguyên.

    Input:  {'gender': 'Male', 'Contract': 'Month-to-month', 'tenure': 12, ...}
    Output: {'gender': 1, 'Contract': 0, 'tenure': 12, ...,
             'ChargePerTenure': ..., 'NumServices': ...}
    """
    encoded = {}
    for key, val in customer_raw.items():
        if key in ENCODING_MAP:
            encoded[key] = ENCODING_MAP[key].get(str(val), 0)
        else:
            encoded[key] = val

    # Feature engineering — giống hệt notebook
    tenure  = encoded.get('tenure', 1)
    monthly = encoded.get('MonthlyCharges', 0)
    encoded['ChargePerTenure'] = monthly / (tenure + 1)
    encoded['NumServices']     = sum(encoded.get(c, 0) for c in SERVICE_COLS)

    return encoded


def preprocess_single(customer_raw: dict, scaler):
    """
    Encode + scale 1 khách hàng.
    Trả về (X_scaled: np.ndarray, X_df: DataFrame)
    """
    encoded = encode_single(customer_raw)
    X_df     = pd.DataFrame([encoded])[FEATURE_ORDER]
    X_scaled = scaler.transform(X_df)
    return X_scaled, X_df


def preprocess_batch(df_raw: pd.DataFrame, scaler):
    """
    Xử lý DataFrame nhiều khách hàng (cùng cấu trúc Telco CSV).
    Trả về (X_scaled: np.ndarray, X_df: DataFrame)
    """
    df = df_raw.copy()

    # Fix TotalCharges (có thể là string do khoảng trắng)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Bỏ cột không cần
    for col in ['customerID', 'Churn']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode tất cả categorical
    for col, mapping in ENCODING_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Feature engineering
    df['ChargePerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    service_present       = [c for c in SERVICE_COLS if c in df.columns]
    df['NumServices']     = df[service_present].sum(axis=1)

    X_df     = df[FEATURE_ORDER]
    X_scaled = scaler.transform(X_df)
    return X_scaled, X_df


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict_single(model, scaler, customer_raw: dict):
    """
    Dự đoán 1 khách hàng.

    Returns:
        prob    (float)     — xác suất churn [0, 1]
        label   (str)       — 'CHURN' hoặc 'KHÔNG CHURN'
        X_scaled (ndarray)  — dữ liệu đã scale
        X_df    (DataFrame) — dữ liệu đã encode (chưa scale, dùng cho SHAP)
    """
    X_scaled, X_df = preprocess_single(customer_raw, scaler)
    prob  = model.predict_proba(X_scaled)[0][1]
    label = 'CHURN' if prob > 0.5 else 'KHÔNG CHURN'
    return prob, label, X_scaled, X_df


def predict_batch(model, scaler, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Dự đoán hàng loạt từ DataFrame.

    Returns: DataFrame gốc + cột Churn_Probability, Prediction, Risk_Level
             đã sắp xếp theo nguy cơ giảm dần.
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
# HELPER
# ─────────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple:
    """
    Trả về (emoji, label, màu hex) tương ứng với xác suất.
    Dùng để render badge và màu sắc trong UI.
    """
    if prob >= 0.7:
        return '🔴', 'Nguy cơ CAO', '#E8593C'
    elif prob >= 0.4:
        return '🟡', 'Nguy cơ TRUNG BÌNH', '#f39c12'
    else:
        return '🟢', 'Nguy cơ THẤP', '#1D9E75'


def get_vi_feature_names(feature_list: list) -> list:
    """Chuyển feature names sang tiếng Việt để hiển thị trong SHAP plot."""
    return [FEATURE_LABELS.get(f, f) for f in feature_list]
