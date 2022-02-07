FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime


RUN mkdir /sensorsbp
COPY ./ /sensorsbp
WORKDIR /sensorsbp

RUN apt-get undate && apt-get install git

# pip install
RUN pip install Cython==0.29.26
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/cainmagi/MDNC.git


