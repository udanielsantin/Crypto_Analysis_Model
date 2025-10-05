import json
import boto3
import pandas as pd
import datetime
import threading
import time
from websocket import WebSocketApp

# ========================
# CONFIGURAÇÕES GERAIS
# ========================
BUCKET_NAME = "binance-websocket-stream-data"
SYMBOL = "btcusdt"  # Par de moedas a ser coletado
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"

# Cliente S3
s3 = boto3.client("s3")

# Buffer para armazenar trades temporariamente
trade_buffer = []
lock = threading.Lock()

# Intervalo de upload (em segundos)
UPLOAD_INTERVAL = 5 * 60  # 5 minutos


# ========================
# FUNÇÃO PARA SALVAR NO S3
# ========================
def upload_buffer_to_s3():
    global trade_buffer

    while True:
        time.sleep(UPLOAD_INTERVAL)
        with lock:
            if not trade_buffer:
                continue

            # Cria DataFrame e limpa o buffer
            df = pd.DataFrame(trade_buffer)
            trade_buffer = []

        timestamp = datetime.datetime.utcnow()
        date_path = timestamp.strftime("%Y/%m/%d") 
        time_str = timestamp.strftime("%H-%M-%S")
        key = f"btc-trades/{date_path}/{time_str}.parquet"

        # Converte para Parquet em memória
        parquet_bytes = df.to_parquet(index=False, engine="pyarrow")

        # Envia para o S3
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_bytes)
        print(f"✅ {len(df)} registros enviados para {key}")


# ========================
# CALLBACKS DO WEBSOCKET
# ========================
def on_message(ws, message):
    data = json.loads(message)
    trade = {
        "event_time": datetime.datetime.utcfromtimestamp(data.get("E", 0) / 1000),
        "symbol": data.get("s", ""),
        "price": float(data.get("p", 0)),
        "quantity": float(data.get("q", 0)),
        "buyer_order_id": data.get("b", None),
        "seller_order_id": data.get("a", None),
        "trade_time": datetime.datetime.utcfromtimestamp(data.get("T", 0) / 1000),
        "is_buyer_maker": data.get("m", None),
    }

    # Adiciona no buffer com lock (thread-safe)
    with lock:
        trade_buffer.append(trade)


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
    # Thread para upload periódico
    uploader_thread = threading.Thread(target=upload_buffer_to_s3, daemon=True)
    uploader_thread.start()

    # Conecta ao WebSocket
    ws = WebSocketApp(
        STREAM_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()
