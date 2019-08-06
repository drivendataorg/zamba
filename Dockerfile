# Dockerfile for CI on codeship
FROM continuumio/anaconda3:5.2.0
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y build-essential

RUN apt-get install -y --allow-unauthenticated ffmpeg
RUN apt-get install -y unzip
RUN pip install -U pip Cython

# download weights
RUN mkdir -p /app/zamba/models/cnnensemble
RUN wget https://s3.amazonaws.com/drivendata-public-assets/zamba.zip -P /tmp && \
	unzip /tmp/zamba.zip -d /app/zamba/models/cnnensemble
RUN wget https://s3.amazonaws.com/drivendata-public-assets/data_fast.zip -P /tmp && \
	unzip /tmp/data_fast.zip -d /app/zamba/models/cnnensemble
RUN wget https://s3.amazonaws.com/drivendata-public-assets/input.tar.gz -P /tmp && \
	tar xvzf /tmp/input.tar.gz -C /app/zamba/models/cnnensemble/

RUN mkdir -p ~/.keras/models
RUN wget https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile-no-top.h5 \
	-O /root/.keras/models/nasnet_mobile_no_top.h5
RUN wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 \
	-O /root/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
RUN wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  \
	-O /root/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5


COPY requirements-dev.txt requirements.txt /app/
COPY zamba/models/cnnensemble/requirements.txt /app/zamba/models/cnnensemble
RUN mkdir /app/docs
COPY docs/requirements.txt /app/docs

WORKDIR /app
RUN pip install -r requirements-dev.txt

COPY . /app
RUN pip install .[cpu]
