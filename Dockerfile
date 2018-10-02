# Dockerfile for CI on codeship
FROM continuumio/anaconda3:5.2.0
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y build-essential

RUN apt-get install -y --allow-unauthenticated ffmpeg
RUN pip install -U pip Cython

COPY . /app
WORKDIR /app

RUN pip install -r requirements-dev.txt

RUN pip install .[cpu]
