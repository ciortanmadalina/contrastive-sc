FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
RUN apt update \
    && apt install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*
RUN pip install setuptools==45.0.0 \
    jupyterlab \
    notebook \
    sklearn \
    lmdb \
    attrdict \
    h5py \
    scipy==1.6.0 \
    ipywidgets \
    keras \
    tensorflow-gpu==1.15.0 \
    tensorboard==1.15.0 \
    tensorboardX \
    leidenalg \
    scikit-learn==0.22.2 \
    scanpy==1.5.1 \
    jgraph \
    louvain \
    xlrd \
    openpyxl 

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

EXPOSE 8080 8888 6006

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
