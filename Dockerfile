FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /opt/triton-nautilus/

COPY . .

RUN pip install .
