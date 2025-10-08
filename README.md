# ⚡ CRYPTO ANALYSIS MODEL

Este projeto implementa um **pipeline completo de análise e previsão de preços de criptomoedas** (Bitcoin) utilizando dados em tempo real da **Binance**.  
O sistema é totalmente automatizado em **AWS**, com coleta contínua de dados, **treinamento de modelo de Machine Learning** e **dashboard interativo** para visualização das previsões.

---

## 🚀 Arquitetura Geral

A solução é composta por três principais componentes:

1. **📡 Coletor (EC2 #1 - binance-collector)**  
   - Conecta-se ao **WebSocket da Binance** para receber *trades* em tempo real.  
   - Agrupa os dados **por minuto**.  
   - Salva os dados em formato **Parquet** no **Amazon S3**, particionados por:
     ```
     s3://binance-websocket-stream-data/year=YYYY/month=MM/day=DD/hour=HH/minute=MM/
     ```

2. **🧠 Treinamento de Modelo (EC2 #2 - predictions_model)**  
   - Lê os dados brutos do bucket S3.  
   - Realiza pré-processamento, engenharia de features e treinamento de um modelo de Machine Learning (ex: Random Forest).  
   - O modelo treinado é salvo em outro bucket S3:
     ```
     s3://binance-websocket-stream-data/btc-models/
     ```
   - Cada modelo é versionado com o timestamp:
     ```
     btc_price_model_YYYY-MM-DD_HH-MM.joblib
     ```

3. **📊 Dashboard (Streamlit)**  
   - Consome automaticamente o **último modelo disponível** no bucket S3.  
   - Carrega dados de entrada recentes e exibe previsões em tempo real.  
   - Mantém-se online 24/7 (ex: via Render, EC2 ou Railway).

---

## 🧩 Estrutura do Projeto
Crypto_Analysis_Model/  
│  
├── binance-collector/ # Coleta de dados via WebSocket e envio ao S3  
├── predictions_model/ # Código de treino e salvamento do modelo  
├── dashboard/ # Dashboard Streamlit para visualização das previsões  
│  
├── requirements.txt # Dependências Python  
├── render.yaml # Configuração de deploy Render  
├── start.sh # Script de inicialização do Streamlit  
└── README.md  
