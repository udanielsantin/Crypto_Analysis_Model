import json
import boto3
import pandas as pd
import datetime
import threading
from websocket import WebSocketApp

# ========================
# CONFIGURAÇÕES GERAIS
# ========================
BUCKET_NAME = "binance-websocket-stream-data"
SYMBOL = "btcusdt"  # Par de moedas a ser coletado
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"

# Cliente S3
s3 = boto3.client("s3")

# ========================
# FUNÇÃO PARA SALVAR NO S3
# ========================
def save_to_s3(message):
    data = json.loads(message)
    df = pd.DataFrame([{
        "event_time": datetime.datetime.utcfromtimestamp(data.get("E", 0) / 1000),
        "symbol": data.get("s", ""),
        "price": float(data.get("p", 0)),
        "quantity": float(data.get("q", 0)),
        "buyer_order_id": data.get("b", None),
        "seller_order_id": data.get("a", None),
        "trade_time": datetime.datetime.utcfromtimestamp(data.get("T", 0) / 1000),
        "is_buyer_maker": data.get("m", None),
    }])

    # Gera o nome do arquivo com timestamp
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")
    key = f"btc-trades/{timestamp}.parquet"

    # Converte pra Parquet em memória
    parquet_bytes = df.to_parquet(index=False, engine="pyarrow")

    # Envia para o S3
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_bytes)

    print(f"✅ Enviado: {key}")


# ========================
# CALLBACKS DO WEBSOCKET
# ========================
def on_message(ws, message):
    # Roda salvamento em thread separada pra não travar o stream
    threading.Thread(target=save_to_s3, args=(message,)).start()

def on_error(ws, error):
    print(f"❌ Erro: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔒 Conexão fechada com a Binance")

def on_open(ws):
    print("🔌 Conectado no WebSocket Binance")


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    ws = WebSocketApp(
        STREAM_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()
