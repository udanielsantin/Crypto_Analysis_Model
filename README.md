# CRYPTO ANALYSIS MODEL

Este projeto coleta dados em tempo real da Binance via WebSocket,
agrupa os trades por minuto e salva no Amazon S3 em formato **Parquet**,
já particionado por `year/month/day/hour/minute`.

## 🚀 Como rodar na AWS EC2

1. Clone o repositório:
   ```bash
   git clone https://github.com/<seu-usuario>/binance-stream-ml.git
   cd binance-stream-ml
