FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime


RUN mkdir /sensorsbp
COPY ./ /sensorsbp
WORKDIR /sensorsbp

# pip install
RUN pip install -r requirements.txt


