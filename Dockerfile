# Python 3.11のスリムイメージを使用
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /app/src && \
    mkdir -p /app/data

# アプリケーションコードをコピー
COPY src/ ./src/
COPY data/ ./data/

# Pythonパスを設定
ENV PYTHONPATH=/app/src

# ポート8000を公開（必要に応じて変更）
EXPOSE 8000

# デフォルトコマンド（アプリケーションの起動方法に応じて変更）
# CMD ["python", "-m", "src.generator.analysis_data_generator"] 