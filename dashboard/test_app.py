# dashboard/test_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
from datetime import datetime
import os
import matplotlib.pyplot as plt

st.title("BTC Price Prediction - 5 Minutos")

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
    # st.success(f"Modelo carregado do S3: {key}")
    return model

model = load_latest_model(S3_BUCKET, S3_PREFIX)

# ========================
# CARREGA OS ÚLTIMOS DADOS DO S3
# ========================
# Define o prefixo de dados do S3 de acordo com a data atual
today = datetime.utcnow()
S3_DATA_PREFIX = f"btc-trades2/{today.year:04d}/{today.month:02d}/{today.day:02d}/"
N = 1000  # número de trades para carregar
N_DISPLAY = 100  # número de trades para exibir na tabela

def load_latest_parquet(bucket, prefix, n):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in objs:
        st.error("Nenhum arquivo de dados encontrado no S3")
        return None

    parquet_objs = sorted(
        [obj for obj in objs["Contents"] if obj["Key"].endswith(".parquet")],
        key=lambda x: x["Key"],
        reverse=True
    )
    if not parquet_objs:
        st.error("Nenhum arquivo parquet encontrado")
        return None

    latest_key = parquet_objs[0]["Key"]
    response = s3.get_object(Bucket=bucket, Key=latest_key)
    df = pd.read_parquet(BytesIO(response["Body"].read()))
    df = df.sort_values("trade_time").tail(n)
    return df

df = load_latest_parquet(S3_BUCKET, S3_DATA_PREFIX, N)
if df is not None:
    df_display = df.tail(N_DISPLAY)
    st.write(f"Exibindo os últimos {N_DISPLAY} trades:")
    st.dataframe(df_display)

    # Agrupa por ano-mês-dia-hora-minuto-segundo usando event_time
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["event_time_group"] = df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    price_by_time = df.groupby("event_time_group")["price"].mean().reset_index()
    # Gráfico simples e bonito
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(price_by_time["event_time_group"], price_by_time["price"], color="#0074D9", linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Ano-Mês-Dia Hora:Minuto:Segundo", fontsize=12)
    ax.set_ylabel("Preço (USD)", fontsize=12)
    ax.set_title("Variação do Preço BTC (últimos 100 trades)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, fontsize=10)
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True, linestyle='--', alpha=0.5)
    price_min = price_by_time["price"].min()
    price_max = price_by_time["price"].max()
    margin = (price_max - price_min) * 0.05  # aumentou a margem para 5%
    ax.set_ylim(price_min - margin, price_max + margin)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Não foi possível carregar os dados de trades.")

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
