import json
import boto3
import pandas as pd
from websocket import WebSocketApp
from datetime import datetime
import threading

# Nome do bucket S3 (troca se for diferente)
BUCKET_NAME = "binance-websocket-stream-data"
s3 = boto3.client("s3")

# Buffer de trades em memória
trade_buffer = []
lock = threading.Lock()

def save_to_s3():
    """Acumula trades e salva a cada minuto em Parquet no S3"""
    global trade_buffer

    while True:
        now = datetime.utcnow()
        # Próximo minuto cheio
        next_minute = (now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1))
        wait_seconds = (next_minute - datetime.utcnow()).total_seconds()

        threading.Event().wait(wait_seconds)  # espera até virar o minuto

        with lock:
            if trade_buffer:
                df = pd.DataFrame(trade_buffer)
                trade_buffer = []

                # Caminho no S3 organizado por partição
                key = (
                    f"trades/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
                    f"hour={now.hour:02d}/minute={now.minute:02d}/trades.parquet"
                )

                # Salva em Parquet
                parquet_bytes = df.to_parquet(index=False, engine="pyarrow")
                s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_bytes)

                print(f"✅ Arquivo salvo no S3: {key} | {len(df)} trades")

def on_message(ws, message):
    """Recebe trade do WebSocket da Binance"""
    global trade_buffer
    data = json.loads(message)

    trade = {
        "trade_id": data["t"],
        "price": float(data["p"]),
        "qty": float(data["q"]),
        "timestamp": pd.to_datetime(data["T"], unit="ms"),
        "is_buyer_maker": data["m"]
    }

    with lock:
        trade_buffer.append(trade)

def on_open(ws):
    """Abre conexão no WebSocket da Binance"""
    print("🔌 Conectado no WebSocket Binance")
    payload = {"method": "SUBSCRIBE", "params": ["btcusdt@trade"], "id": 1}
    ws.send(json.dumps(payload))

if __name__ == "__main__":
    # Thread que salva os dados no S3 a cada minuto
    saver_thread = threading.Thread(target=save_to_s3, daemon=True)
    saver_thread.start()

    # Conexão WebSocket Binance
    ws = WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@trade",
        on_message=on_message,
        on_open=on_open
    )
    ws.run_forever()
