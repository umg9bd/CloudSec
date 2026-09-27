# Runtime for the real-time pipeline (pipeline.py) and every torch-based script.
# Pinned to the versions the checkpoints were trained/evaluated with. CPU-only torch.
#
#   docker build -t cloudsec .
#   docker run --rm -v "$PWD:/app" cloudsec python pipeline.py --watch incoming
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install torch-geometric==2.8.0.post1 numpy==2.5.1 pandas==3.0.3 pyarrow==25.0.0 \
        scipy==1.18.1 scikit-learn==1.9.0 xgboost==3.4.1 networkx==3.6.1 policy_sentry==0.15.2 \
        neo4j==6.2.0 PyYAML watchdog tqdm python-dateutil

WORKDIR /app
