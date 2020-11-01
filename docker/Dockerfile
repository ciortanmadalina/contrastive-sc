# docker run --runtime nvidia -it --shm-size 8G -v ~/notebooks:/workspace/notebooks  -v /datasets/asterion:/workspace/datasets -p 8888:8888 -p 6006:6006 -p 6007:6007 asterion/contextual

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
#FROM asterion/parlai
#RUN apt install nodejs npm
RUN apt update \
    && apt install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*
RUN pip install jupyterlab \
    notebook \
    sklearn \
    lmdb \
    attrdict \
    h5py \
    scipy==1.1.0 \
    pillow==6.1 \
    ipywidgets \
    hyperopt \
    keras \
    opencv-python \
    tensorflow-gpu==1.15.0 \
    tensorboard==1.15.0 \
    tensorboardX \
    leidenalg \
    scikit-learn==0.22.2 \
    scanpy==1.5.1 \
    jgraph \
    igraph \
    louvain \
    xlrd \
    openpyxl 



#RUN git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
#RUN cd ~/ParlAI; python setup.py develop
# RUN cd ~/ParlAI; python projects/personality_captions/interactive.py -mf models:personality_captions/transresnet/model

# CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]

# Jupyter notebook configuration
RUN pip install yapf \
    jupyter_contrib_nbextensions \
    jupyter_highlight_selected_word
    
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
# RUN pip install jupyter-tensorboard
# RUN jupyter tensorboard enable --user

RUN jupyter contrib nbextension install --user
RUN jupyter nbextension install https://github.com/jfbercher/code_prettify/archive/master.zip --user
RUN jupyter nbextension enable code_prettify-master/code_prettify
RUN jupyter nbextension install --py jupyter_highlight_selected_word
RUN jupyter nbextension enable highlight_selected_word/main

EXPOSE 8080 8888 6006 6007

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
