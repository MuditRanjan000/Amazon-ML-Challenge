# Reproducible runtime for AWS / any Linux box. Data is mounted, never baked in:
#   docker build -t er .
#   docker run --rm -v /data/dataset:/data -v $PWD/output:/app/output er \
#       python execution/run_blocking_eval.py --exp-id BLK-100 --direction both --save
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 ER_DATA_DIR=/data ER_OUTPUT_DIR=/app/output
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY execution ./execution
COPY scripts ./scripts
COPY experiments ./experiments
COPY tests ./tests
RUN pip install --no-cache-dir --no-deps -e .
CMD ["python", "-m", "pytest", "-q"]
