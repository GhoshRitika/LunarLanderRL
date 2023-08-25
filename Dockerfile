ARG PYTHON_VERSION=3.9
FROM python:$PYTHON_VERSION

RUN apt-get update && apt-get upgrade -y 

# Install packages
RUN apt-get install -y wget git vim libsm6 libxext6 libxrender-dev ffmpeg python-opengl

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN bash Anaconda3-2019.10-Linux-x86_64.sh -b
RUN rm Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
RUN conda update conda
RUN yes | conda update anaconda
RUN yes | conda update --all
RUN conda init

# Install packages
RUN conda install -y -c pytorch pytorch torchvision
RUN conda install -y tensorflow-gpu==1.15.0
RUN pip install gin-config==0.4.0
RUN pip install gym==0.17.3
RUN pip install gym[box2d]

# Add a directory for python packages to be mounted
ENV PYTHONPATH /root/pkgs:$PYTHONPATH

RUN apt-get install -y freeglut3-dev
RUN conda install -y PyOpenGL
RUN pip install pygame PyOpenGL_accelerate

WORKDIR /root/pkgs
#fork of the original dl repo with edited 
RUN git clone --branch ritika https://github.com/GhoshRitika/dl.git
# WORKDIR /root/pkgs/dl
# RUN git checkout 650db8abc90053305be95e73ce28da624e9092dc

WORKDIR /root
# Bash entrypoint
ENTRYPOINT /bin/bash
