# ======================================================
# dashboard.py - Dashboard de previsão BTC
# ======================================================

from flask import Flask, render_template_string
import boto3
import pandas as pd
import joblib
import io
import datetime

# ============================
# CONFIGURAÇÕES GERAIS
# ============================
BUCKET_NAME = "binance-websocket-stream-data"
MODEL_PREFIX = "btc-models/"
TRADES_PREFIX = "btc-trades2/"
s3 = boto3.client("s3")

app = Flask(__name__)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def get_latest_model_key():
    objs = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=MODEL_PREFIX)
    models = [obj["Key"] for obj in objs.get("Contents", []) if obj["Key"].endswith(".joblib")]
    if not models:
        return None
    return sorted(models)[-1]  # mais recente

def load_latest_model():
    key = get_latest_model_key()
    if not key:
        return None, None
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    model = joblib.load(io.BytesIO(obj["Body"].read()))
    return model, key.split("/")[-1]

def load_recent_trades(limit=200):
    objs = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=TRADES_PREFIX)
    keys = [o["Key"] for o in objs.get("Contents", []) if o["Key"].endswith(".parquet")]
    if not keys:
        return None
    keys = sorted(keys)[-3:]  # últimos arquivos
    dfs = []
    for k in keys:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=k)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        dfs.append(df)
    df_all = pd.concat(dfs).tail(limit)
    return df_all

def predict_next_move(model, df):
    if df is None or len(df) < 10:
        return "Sem dados"
    df["price_change"] = df["price"].pct_change()
    df["rolling_mean"] = df["price"].rolling(10).mean()
    df["rolling_std"] = df["price"].rolling(10).std()
    df = df.dropna()
    X = df[["price_change", "rolling_mean", "rolling_std"]].tail(1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return ("Subida 🚀" if pred == 1 else "Descida 📉"), round(prob * 100, 2)

# ============================
# ROTAS
# ============================

@app.route("/")
def index():
    model, model_name = load_latest_model()
    trades = load_recent_trades()

    if model is None:
        return "<h2>Nenhum modelo encontrado no S3</h2>"

    decision, prob = predict_next_move(model, trades)

    last_price = trades["price"].iloc[-1] if trades is not None else 0

    html = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="30">
        <title>BTC Dashboard</title>
        <style>
            body {{ font-family: Arial; background: #111; color: #fff; text-align: center; }}
            h1 {{ color: #0f0; }}
            .card {{ margin: 30px auto; padding: 20px; width: 400px; border-radius: 12px; background: #222; }}
        </style>
    </head>
    <body>
        <h1>BTC Dashboard - Previsão</h1>
        <div class="card">
            <p><b>Modelo:</b> {model_name}</p>
            <p><b>Último preço:</b> {last_price:.2f}</p>
            <p><b>Previsão (5 min):</b> {decision}</p>
            <p><b>Confiança:</b> {prob}%</p>
            <p><small>Atualizado em: {datetime.datetime.now().strftime("%H:%M:%S")}</small></p>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
