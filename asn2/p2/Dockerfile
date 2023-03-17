FROM python:3.8

WORKDIR /app

# RUN --mount=type=cache,target=/root/.cache \
#     pip3 install sentence-transformers transformers 
RUN --mount=type=cache,target=/root/.cache \
    pip install git+https://github.com/huggingface/transformers.git

RUN --mount=type=cache,target=/root/.cache \
    pip install torch

CMD ["/bin/sh", "-c", "/app/main.sh > A2Part2.txt"]