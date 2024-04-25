# Langchain Services

## 1. RAG (Retrieval-Augmented Generation) service

### 1.1. Donwload data

Require **wget** and **gdown** package

```bash
$ cd data_source/generative_ai && python download.py
$ cd ../generative_ai && python download.py
```

### 1.2. Run service

```bash
$ pip3 install -r dev_requirements.txt
$ unicorn rag.main:app --host "0.0.0.0" --port 5000 --reload
```
