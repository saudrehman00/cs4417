FROM python:3.7

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache \
    pip install sentence-transformers transformers torch tensorflow
    # !pip install -q git+https://github.com/huggingface/transformers.git 

CMD ["/bin/sh", "-c", "/app/main.sh"]
