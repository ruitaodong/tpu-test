# TorchTPU on minimal Ubuntu 24.04 (noble)
#
# Noble ships Python 3.12, which torch_tpu supports (it publishes cp312 wheels),
# so we use the distro's own Python and skip uv entirely. torch_tpu 0.2.0 wheels
# are tagged manylinux_2_31; noble's glibc 2.39 satisfies that easily.
#
# Build:  docker build -t tpu-test .
# Run on a TPU VM:  sudo docker run --rm --privileged --net=host tpu-test

############################
# Stage 1: build the venv
############################
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

# A venv sidesteps noble's "externally managed" system Python (PEP 668)
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# CPU-only torch from the PyTorch index (no CUDA libraries, much smaller image);
# torch_tpu itself comes from PyPI.
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch_tpu==0.2.0

# Add your serving dependencies here, e.g.:
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt

############################
# Stage 2: minimal runtime
############################
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# Same distro Python the venv was built against (no pip/venv tooling needed here)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# COPY . /app

# Default command: smoke test that the container can see the TPU.
# Replace with your server, e.g. CMD ["python", "serve.py"]
CMD ["python", "-c", "import torch, torch_tpu; x = torch.ones(2, 2, device='tpu'); print('TPU OK:', (x + 1).cpu())"]
