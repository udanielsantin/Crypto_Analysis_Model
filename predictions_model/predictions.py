import awswrangler as wr
import pandas as pd
from datetime import datetime, timedelta
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import threading
from io import BytesIO
import boto3

# ========================
# CONFIGURAÇÕES
# ========================
BUCKET_NAME = "binance-websocket-stream-data"
TRADES_PREFIX = "btc-trades2/"
MODEL_PREFIX = "btc-models/"
RUN_INTERVAL = 5 * 60  # 5 minutos

s3 = boto3.client("s3")

# ========================
# FUNÇÃO PARA CARREGAR TRADES
# ========================
def load_recent_trades():
    today = datetime.utcnow()
    yesterday = today - timedelta(days=1)

    # Filtra os dois dias mais recentes
    partition_filter = lambda x: (
        x["year"] == str(yesterday.year) and x["month"] == f"{yesterday.month:02}" and x["day"] in [f"{yesterday.day:02}", f"{today.day:02}"]
    )

    try:
        df = wr.s3.read_parquet(
            path=f"s3://{BUCKET_NAME}/{TRADES_PREFIX}",
            dataset=True,
            partition_filter=partition_filter
        )
        df = df.sort_values("trade_time")
        print(f"✅ {len(df)} trades carregados do S3")
        return df
    except Exception as e:
        print(f"⚠️ Nenhum trade carregado: {e}")
        return pd.DataFrame()

# ========================
# FUNÇÃO DE FEATURE ENGINEERING
# ========================
def create_features(df):
    df["price_diff"] = df["price"].diff()
    df["price_up"] = (df["price_diff"] > 0).astype(int)

    # Features rolling 5 trades
    df["price_mean_5"] = df["price"].rolling(5).mean()
    df["price_std_5"] = df["price"].rolling(5).std()
    df["quantity_mean_5"] = df["quantity"].rolling(5).mean()

    df = df.dropna()
    return df

# ========================
# FUNÇÃO PARA TREINAR MODELO
# ========================
def train_model(df):
    X = df[["price_diff", "price_mean_5", "price_std_5", "quantity_mean_5"]]
    y = df["price_up"]

    if len(df) < 5:
        print(f"⚠️ Dados muito poucos ({len(df)} linhas) para treinar modelo confiável. Pulando.")
        return None

    test_size = 0.2
    if len(df) * test_size < 1:
        test_size = 1 / len(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model

# ========================
# FUNÇÃO PARA SALVAR MODELO NO S3
# ========================
def save_model_to_s3(model):
    if model is None:
        return
    timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M')
    key = f"{MODEL_PREFIX}btc_price_model_{timestamp}.joblib"
    model_bytes = BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=model_bytes.getvalue())
    print(f"✅ Modelo salvo em s3://{BUCKET_NAME}/{key}")

# ========================
# LOOP PRINCIPAL
# ========================
def run_loop():
    while True:
        try:
            print(f"⏱️ Iniciando ciclo de treino - {datetime.utcnow()}")
            df = load_recent_trades()
            
            if df.empty:
                print("⚠️ Nenhum trade encontrado para treinamento. Pulando ciclo.")
            else:
                df = create_features(df)
                model = train_model(df)
                save_model_to_s3(model)

        except Exception as e:
            print(f"❌ Erro no ciclo: {e}")

        time.sleep(RUN_INTERVAL)

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    run_thread = threading.Thread(target=run_loop, daemon=True)
    run_thread.start()
    run_thread.join()
