FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir \
        scikit-learn==1.3.2 \
        pandas==2.2.2 \
        joblib==1.4.2 \
        mlflow-skinny==3.11.1

COPY src/ /opt/ml/code/

# SageMaker invokes the container with the command `train`.
RUN printf '#!/bin/sh\nexec python /opt/ml/code/train.py "$@"\n' > /usr/local/bin/train \
 && chmod +x /usr/local/bin/train
