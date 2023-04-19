FROM ubuntu:20.04

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV TZ Australia/Melbourne

RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gnupg \
    jq \
    lsb-release \
    pkg-config \
    software-properties-common \
    python3-distutils \
    python3-pip

# require min git version 2.18 - ubuntu latest has 2.17.1
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git

# install Python 3.7 - use default Python 3.8
# RUN add-apt-repository ppa:deadsnakes/ppa -y && \
#     apt-get update && \
#     apt-get install python3.7 python3-pip python3.7-dev -y && \
#     rm /usr/bin/python3 && \
#     ln -s python3.7 /usr/bin/python3

# cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install Cython==0.29.* && \
    python3 -m pip install -r requirements.txt

COPY ./gpt4all_api/ ./gpt4all_api/

ENV PYTHONPATH=gpt4all_api

RUN mkdir -p /tmp/offload

CMD ["python3", "./gpt4all_api/app.py"]
