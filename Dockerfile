FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/data/manuals /app/chroma_db

COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8080
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["/start.sh"]