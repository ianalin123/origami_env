FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
&& pip install --no-cache-dir "openenv-core[core]>=0.2.1"

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
