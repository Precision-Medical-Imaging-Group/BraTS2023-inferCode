FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN apt-get -y update
RUN apt-get -y install git

COPY . /mlcube_project/
RUN bash /mlcube_project/install_nnunet.sh
RUN pip3 install --no-cache-dir -r /mlcube_project/requirements.txt
ENV LANG C.UTF-8
WORKDIR /mlcube_project/
ENTRYPOINT ["python3", "mlcube.py"]