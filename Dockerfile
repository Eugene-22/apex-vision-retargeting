FROM rysen_sdk:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    wget \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python-headless \
    mediapipe \
    pytest

WORKDIR /workspace/apex-vision-retargeting

CMD ["/bin/bash"]
