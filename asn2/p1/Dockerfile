FROM python:3.8

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache \
    pip3 install sentence-transformers
    # !pip install git+https://github.com/huggingface/transformers.git 
    
CMD ["/bin/sh", "-c", "/app/main.sh > A2Part1.txt"]