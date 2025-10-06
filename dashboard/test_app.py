# dashboard/test_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
from datetime import datetime
import os

st.title("BTC Price Prediction - 5 Minutos (Teste)")

# ========================
# CONFIGURAÇÃO AWS
# ========================
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = "binance-websocket-stream-data"
S3_PREFIX = "btc-models/"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# ========================
# FUNÇÃO PARA CARREGAR MODELO DO S3 COM CACHE
# ========================
@st.cache_data(ttl=300)  # Atualiza a cada 5 minutos
def load_latest_model(bucket, prefix):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in objs:
        st.error("Nenhum modelo encontrado no S3")
        return None

    latest_obj = max(objs["Contents"], key=lambda x: x["LastModified"])
    key = latest_obj["Key"]

    response = s3.get_object(Bucket=bucket, Key=key)
    model_bytes = response["Body"].read()
    model = joblib.load(BytesIO(model_bytes))
    st.success(f"Modelo carregado do S3: {key}")
    return model

model = load_latest_model(S3_BUCKET, S3_PREFIX)

# ========================
# SIMULAÇÃO DE TRADES
# ========================
N = 20
now = pd.Timestamp.now()
df = pd.DataFrame({
    "trade_time": [now - pd.Timedelta(seconds=5*i) for i in range(N)][::-1],
    "price": np.random.normal(50000, 50, N),
    "quantity": np.random.uniform(0.001, 0.01, N)
})

st.write(f"Total de trades simulados: {len(df)}")
st.dataframe(df.tail(10))

# ========================
# GERA FEATURES SIMPLES
# ========================
df = df.sort_values("trade_time")
df["price_mean_5"] = df["price"].rolling(5).mean()
df["price_std_5"] = df["price"].rolling(5).std()
df["quantity_mean_5"] = df["quantity"].rolling(5).mean()

feature_cols = ["price_mean_5", "price_std_5", "quantity_mean_5"]  
X_features = df[feature_cols].tail(1).fillna(0).to_numpy()


# ========================
# FAZ PREVISÃO
# ========================
if model is not None:
    prediction = model.predict(X_features)[0]
    prob = model.predict_proba(X_features)[0]

    if prediction == 1:
        st.success(f"➡️ Sinal de **subida** nos próximos 5 minutos! (Confiança: {prob[1]*100:.2f}%)")
    else:
        st.error(f"➡️ Sinal de **descida** nos próximos 5 minutos! (Confiança: {prob[0]*100:.2f}%)")
else:
    st.warning("Modelo não carregado, impossível fazer previsão")
