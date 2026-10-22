# Runtime for the real-time pipeline (pipeline.py) and every torch-based script.
# Pinned to the versions the checkpoints were trained/evaluated with. CPU-only torch.
#
#   docker build -t cloudsec .
#   docker run --rm -v "$PWD:/app" cloudsec python pipeline.py --watch incoming
#   (or just run.cmd / ./run.sh: the whole system -- the CloudSec app (Agent Gate +
#    Live Dashboard) on http://localhost:8503)
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install torch-geometric==2.8.0.post1 numpy==2.5.1 pandas==3.0.3 pyarrow==25.0.0 \
        scipy==1.18.1 scikit-learn==1.9.0 xgboost==3.4.1 networkx==3.6.1 policy_sentry==0.15.2 \
        neo4j==6.2.0 PyYAML watchdog tqdm python-dateutil

# The dashboard (cloudsec_dashboard.py); its own layer so adding it didn't rebuild torch.
# streamlit 1.64 excludes pyarrow 25.0.0, hence the patch upgrade from the line above.
RUN pip install streamlit==1.64.0 pyarrow==25.0.1

# The agent gate (cloudsec.py / agent_console.py): Stage 1 classifier (transformers; the checkpoint
# itself is mounted from prompt_injection_gate/artifacts/), Stage 2 schemas, Stage 3 pydantic models.
# Pinned to the prompt_injection_gate venv the classifier was trained and evaluated in.
RUN pip install transformers==5.20.1 tokenizers==0.23.3 safetensors==0.8.0 huggingface_hub==1.33.0 \
        sentencepiece==0.2.2 protobuf==7.36.2 pydantic==2.14.1 jsonschema==4.26.0 PyYAML==6.0.3
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CLOUDSEC_IN_DOCKER=1

WORKDIR /app
EXPOSE 8503
