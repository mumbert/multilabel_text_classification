#!/bin/bash

cd data/glove

# Download the GloVe embeddings (6B tokens, 100-dimensional vectors)
wget http://nlp.stanford.edu/data/glove.6B.zip

# Unzip the downloaded file
unzip glove.6B.zip