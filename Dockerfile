FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime


RUN mkdir /bp_benchmark
COPY ./ /bp_benchmark
WORKDIR /bp_benchmark

RUN apt-get update -y && apt-get install git -y && apt-get install tmux -y && apt-get install wget -y && apt-get install unzip -y

# pip install
RUN pip install Cython==0.29.26
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/cainmagi/MDNC.git


