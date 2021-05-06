FROM pytorch/pytorch

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get install -y wget && \
    apt-get install make && \
    apt-get install -y git


WORKDIR /workspace/bp-algorithm/code/train/

COPY . .

# pip install
RUN pip install -r requirements.txt


