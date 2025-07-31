#!/bin/bash

echo "🧹 Uninstalling conflicting packages..."
pip uninstall -y langchain langchain-core langsmith langchain-community langchain-chroma

echo "📦 Installing compatible versions..."
pip install \
  langchain==0.1.14 \
  langchain-core==0.1.40 \
  langsmith==0.1.17 \
  langchain-community==0.0.31 \
  langchain-chroma==0.1.0
