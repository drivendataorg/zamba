# Dockerfile for CI on codeship
FROM continuumio/anaconda3
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y build-essential

RUN conda update -n base conda
RUN conda install av==0.3.3 -c conda-forge
RUN pip install -U pip Cython

COPY . /app
WORKDIR /app

RUN pip install .[tf]
