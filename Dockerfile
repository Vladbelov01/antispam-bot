FROM python:3.11-slim

WORKDIR /app

# Убрали apt-get install -y gcc g++ — не хватает места на bothost
# Устанавливаем только необходимое
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

CMD ["python", "bot.py"]
