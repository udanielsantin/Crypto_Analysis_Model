# ======================================================
# prediction.py - Classificação de movimento BTC
# ======================================================

import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import io
import os

# ======================================================
# 1️⃣ Configurar S3
# ======================================================
s3_bucket = "binance-websocket-stream-data"
s3_prefix = "btc-trades/"  # pasta com os Parquet

s3 = boto3.client("s3")

# ======================================================
# 2️⃣ Ler arquivos Parquet do S3
# ======================================================
def load_s3_parquet(bucket, prefix, limit=200):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    df_list = []

    for obj in objs.get("Contents", [])[:limit]:
        key = obj["Key"]
        if key.endswith(".parquet"):
            resp = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            df_list.append(df)

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        print(f"✅ Total de registros carregados: {len(df)}")
        return df
    else:
        raise ValueError("Nenhum arquivo Parquet encontrado no S3!")

df = load_s3_parquet(s3_bucket, s3_prefix)

# ======================================================
# 3️⃣ Verificar e limpar dados
# ======================================================
if "trade_time" not in df.columns:
    raise ValueError("Coluna 'trade_time' não encontrada nos dados!")

df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["quantity"] = pd.to_numeric(df.get("quantity", 1), errors="coerce")

df = df.dropna(subset=["trade_time", "price"])
df = df.sort_values("trade_time").reset_index(drop=True)

print(f"📅 Período: {df['trade_time'].min()} → {df['trade_time'].max()}")

# ======================================================
# 4️⃣ Criar candles (testar granularidade)
# ======================================================
for freq in ["1min", "30s", "10s", "5s"]:
    df_candle = df.resample(freq, on="trade_time").agg({
        "price": "ohlc",
        "quantity": "sum"
    }).dropna()

    if len(df_candle) > 10:
        print(f"✅ {len(df_candle)} candles gerados com frequência '{freq}'")
        break

if df_candle.empty:
    raise ValueError("Nenhum candle válido gerado! Verifique timestamps ou frequência de trades.")

df_candle.columns = ["open", "high", "low", "close", "volume"]

# ======================================================
# 5️⃣ Criar features técnicas
# ======================================================
df_candle["ma_5"] = df_candle["close"].rolling(5).mean()
df_candle["ma_10"] = df_candle["close"].rolling(10).mean()
df_candle["ma_20"] = df_candle["close"].rolling(20).mean()

df_candle["target"] = (df_candle["close"].shift(-1) > df_candle["close"]).astype(int)
df_candle = df_candle.dropna()

if df_candle.empty:
    raise ValueError("DataFrame vazio depois de criar features e labels!")

# ======================================================
# 6️⃣ Separar treino/teste
# ======================================================
features = ["open", "high", "low", "close", "volume", "ma_5", "ma_10", "ma_20"]
X = df_candle[features]
y = df_candle["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ======================================================
# 7️⃣ Treinar modelo
# ======================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))

# ======================================================
# 8️⃣ Salvar modelo e previsões no S3
# ======================================================
model_file = "binance_movement_model.pkl"
joblib.dump(model, model_file)
s3.upload_file(model_file, s3_bucket, "models/binance_movement_model.pkl")

df_pred = X_test.copy()
df_pred["target"] = y_test
df_pred["prediction"] = y_pred
pred_file = "predictions.csv"
df_pred.to_csv(pred_file, index=False)
s3.upload_file(pred_file, s3_bucket, "predictions/predictions.csv")

print("✅ Modelo e previsões salvos no S3 com sucesso!")
