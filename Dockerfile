FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "pydantic>=2.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0"

COPY . /app

EXPOSE 8000

CMD ["python", "-u", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
