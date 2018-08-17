FROM nvidia/cuda
MAINTAINER canpi

# install anaconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && /bin/bash ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

# install libs
RUN conda install opencv pytorch torchvision && pip install nibabel

# map path
ADD src /
COPY models /models

CMD [ "/bin/bash" ]

