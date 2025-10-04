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
s3_prefix = "btc-trades/"  # pasta com os JSONs

s3 = boto3.client("s3")

# ======================================================
# 2️⃣ Ler arquivos Parquet do S3
# ======================================================
def load_s3_parquet(bucket, prefix):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    df_list = []

    for obj in objs.get("Contents", []):
        key = obj['Key']
        if key.endswith(".parquet"):
            resp = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(resp['Body'].read()))
            df_list.append(df)

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

df = load_s3_parquet(s3_bucket, s3_prefix)

# ======================================================
# 3️⃣ Verificar se tem dados
# ======================================================
if df.empty:
    raise ValueError("Nenhum dado disponível no S3. Verifique os arquivos Parquet!")

# Converter datetime
df['trade_time'] = pd.to_datetime(df['trade_time'])

# Ordenar por tempo
df = df.sort_values('trade_time').reset_index(drop=True)

# ======================================================
# 4️⃣ Criar candles de 1 minuto
# ======================================================
df_candle = df.resample('1min', on='trade_time').agg({
    'price':'ohlc',
    'quantity':'sum'
})

df_candle.columns = ['open','high','low','close','volume']

# Remover candles vazios
df_candle = df_candle.dropna()
if df_candle.empty:
    raise ValueError("Candles vazios depois do resample! Verifique os dados.")

# ======================================================
# 5️⃣ Criar features técnicas
# ======================================================
df_candle['ma_5'] = df_candle['close'].rolling(5).mean()
df_candle['ma_10'] = df_candle['close'].rolling(10).mean()
df_candle['ma_20'] = df_candle['close'].rolling(20).mean()

# Label: 1 se próximo candle sobe, 0 se cai
df_candle['target'] = (df_candle['close'].shift(-1) > df_candle['close']).astype(int)

# Remover linhas com NaN
df_candle = df_candle.dropna()
if df_candle.empty:
    raise ValueError("DataFrame vazio depois de criar features e labels!")

# ======================================================
# 6️⃣ Separar treino/teste
# ======================================================
features = ['open','high','low','close','volume','ma_5','ma_10','ma_20']
X = df_candle[features]
y = df_candle['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ======================================================
# 7️⃣ Treinar RandomForestClassifier
# ======================================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ======================================================
# 8️⃣ Salvar modelo no S3
# ======================================================
model_file = "binance_movement_model.pkl"
joblib.dump(model, model_file)
s3.upload_file(model_file, s3_bucket, "models/binance_movement_model.pkl")

# ======================================================
# 9️⃣ Salvar previsões no S3
# ======================================================
df_pred = X_test.copy()
df_pred['target'] = y_test
df_pred['prediction'] = y_pred
pred_file = "predictions.csv"
df_pred.to_csv(pred_file, index=False)
s3.upload_file(pred_file, s3_bucket, "predictions/predictions.csv")

print("Modelo e previsões salvos no S3 com sucesso!")
