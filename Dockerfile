FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev build-essential curl && rm -rf /var/lib/apt/lists/*

# uv installieren
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

COPY . /app

RUN pip install --upgrade pip setuptools wheel build

RUN uv venv

RUN uv pip install .

CMD ["uv", "run", "main.py"]
