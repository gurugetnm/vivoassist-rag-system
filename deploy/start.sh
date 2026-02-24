#!/usr/bin/env bash
set -e

streamlit run streamlit_app.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true &

nginx -g "daemon off;"