#!/bin/sh

TOOLDIR=$(dirname "$0")
cd "$TOOLDIR"

curl 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' | gunzip > ./train_images
curl 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' | gunzip > ./train_labels
curl 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' | gunzip > ./test_images
curl 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' | gunzip > ./test_labels
