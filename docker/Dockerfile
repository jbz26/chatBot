FROM thuannan/torch-transformer

WORKDIR /backend

COPY ./docker/requirements.txt .
RUN pip install --upgrade wheel setuptools pip \
    && pip install -r requirements.txt --no-cache-dir

RUN pip install rapidocr-onnxruntime

COPY /data_source /backend/data_source
COPY /src /backend/src

WORKDIR /backend
