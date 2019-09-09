# Dockerfile for CI on codeship
FROM python:3.6-stretch
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
	apt-get install -y build-essential && \
	apt-get install -y --allow-unauthenticated ffmpeg && \
	apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev \
	libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
	unzip pkg-config && \
	rm -rf /var/lib/apt/lists/*

# download weights
RUN mkdir -p /root/.cache/zamba/cnnensemble /root/.cache/zamba/megadetector /root/.keras/models && \
	wget https://s3.amazonaws.com/drivendata-public-assets/zamba.zip -P /tmp && \
	unzip -q /tmp/zamba.zip -d /root/.cache/zamba/cnnensemble && \
	wget https://s3.amazonaws.com/drivendata-public-assets/data_fast.zip -P /tmp && \
	unzip -q /tmp/data_fast.zip -d /root/.cache/zamba/cnnensemble && \
	wget https://s3.amazonaws.com/drivendata-public-assets/input.tar.gz -P /tmp && \
	tar xzf /tmp/input.tar.gz -C /root/.cache/zamba/cnnensemble/ && \
	wget https://drivendata-public-assets.s3.amazonaws.com/zamba-and-obj-rec-0.859.joblib \
		-P /root/.cache/zamba/blanknonblank && \
	wget https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/megadetector_v3.pb \
		-P /root/.cache/zamba/megadetector && \
	wget https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile-no-top.h5 \
		-O /root/.keras/models/nasnet_mobile_no_top.h5 && \
	wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5 \
		-P /root/.keras/models && \
	wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
		-P /root/.keras/models && \
	rm /tmp/zamba.zip /tmp/input.tar.gz /tmp/data_fast.zip

RUN mkdir -p /app/zamba/models/cnnensemble /app/docs
COPY requirements-dev.txt requirements.txt /app/
COPY zamba/models/cnnensemble/requirements.txt /app/zamba/models/cnnensemble
COPY docs/requirements.txt /app/docs

WORKDIR /app
RUN pip install -U pip Cython && \
	pip install -r requirements-dev.txt && \
	rm -rf /root/.cache/pip

COPY . /app
RUN pip install .[cpu]
