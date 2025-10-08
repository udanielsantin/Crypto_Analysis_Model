#!/bin/bash
cd /workspaces/Crypto_Analysis_Model/dashboard
streamlit run test_app.py --server.port $PORT --server.address 0.0.0.0
