import numpy as np
import pandas as pd
import joblib

# 28 features chính xác được bóc tách từ object XGBoost của bạn
FEATURE_ORDER = [
    'Client_gender', 'Age', 'Staff_VIB', 'Tenure', 'SMS',
    'Verify_method', 'EB_register_channel', 'No_Activity_Name',
    'Type_Transactions', 'Total_trans_no', 'Avg_Trans_no_month',
    'Avg_Trans_Amount', 'Max_Trans_Amount', 'Min_Trans_Amount',
    'No_CurrentAccount', 'Avg_CurrentAccount_Balance',
    'Max_CurrentAccount_Balance', 'Min_CurrentAccount_Balance',
    'No_TermDeposit', 'Avg_TermDeposit_Balance',
    'Max_TermDeposit_Balance', 'Min_TermDeposit_Balance',
    'No_Loan', 'Avg_Loan_Balance', 'Max_Loan_Balance',
    'Min_Loan_Balance', 'No_CC', 'No_DC'
]

def load_model_bundle(path='best_churn_model.pkl'):
    """
    Do model .pkl hiện tại chỉ chứa object XGBClassifier (không bọc trong dict chứa scaler)
    nên ta chỉ việc load trực tiếp mô hình.
    """
    model = joblib.load(path)
    return model, None

def preprocess_batch(df_raw: pd.DataFrame, scaler=None):
    df = df_raw.copy()
    
    # Đảm bảo dataframe có đủ 28 cột, nếu thiếu thì fill 0
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
            
    X_df = df[FEATURE_ORDER].astype(float)
    return X_df.values, X_df

def preprocess_single(customer_raw: dict, scaler=None):
    df = pd.DataFrame([customer_raw])
    return preprocess_batch(df, scaler)

def predict_single(model, scaler, customer_raw: dict):
    X_scaled, X_df = preprocess_single(customer_raw, scaler)
    prob = model.predict_proba(X_df)[0][1]
    label = 'CHURN' if prob > 0.5 else 'KHÔNG CHURN'
    return prob, label, X_scaled, X_df

def predict_batch(model, scaler, df_raw: pd.DataFrame) -> pd.DataFrame:
    X_scaled, X_df = preprocess_batch(df_raw, scaler)
    probs = model.predict_proba(X_df)[:, 1]
    labels = ['CHURN' if p > 0.5 else 'KHÔNG CHURN' for p in probs]

    result = df_raw.copy()
    result['Churn_Probability'] = np.round(probs, 4)
    result['Prediction'] = labels
    result['Risk_Level'] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['🟢 Thấp', '🟡 Trung bình', '🔴 Cao'],
    )
    return result.sort_values('Churn_Probability', ascending=False).reset_index(drop=True)

def get_risk_level(prob: float) -> tuple:
    if prob >= 0.7:
        return '🔴', 'Nguy cơ CAO', '#E8593C'
    elif prob >= 0.4:
        return '🟡', 'Nguy cơ TRUNG BÌNH', '#f39c12'
    else:
        return '🟢', 'Nguy cơ THẤP', '#1D9E75'