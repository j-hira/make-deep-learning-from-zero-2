FROM continuumio/anaconda3:latest

RUN apt-get update && \
    apt-get install -y nodejs npm && \
    npm install n -g && \
    n stable && \
    apt-get purge -y nodejs npm
