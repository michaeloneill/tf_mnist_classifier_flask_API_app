#  Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import numpy as np
import os
import gzip
import cPickle
from six.moves import urllib
from PIL import Image

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def download(file_name, dir_name, source_url):
    print 'Downloading {}'.format(file_name)
    file_path = os.path.join(dir_name, file_name)
    file_path, _ = urllib.request.urlretrieve(source_url,
                                              file_path)
    return file_path


def extract_labels(f):
    print('Extracting {}'.format(f.name))
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)

        # Convert to one-hot
        y = np.zeros((num_items, 10))
        for i in range(num_items):
            y[i, labels[i]] = 1
            
        return y

    
def extract_images(f):
    print('Extracting {}'.format(f.name))
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1).astype(np.float32)/255.0

        return data


def download_and_extract():
    dir_name = './data/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    local_file = download(TRAIN_IMAGES, dir_name, SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = download(TRAIN_LABELS, dir_name, SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f)
        
    local_file = download(TEST_IMAGES, dir_name, SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = download(TEST_LABELS, dir_name, SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f)

    validation_images = train_images[:5000]
    validation_labels = train_labels[:5000]
    train_images = train_images[5000:]
    train_labels = train_labels[5000:]

    data = [[train_images, train_labels],
            [validation_images, validation_labels],
            [test_images, test_labels]]
    
    with open(dir_name+'mnist.pkl', 'wb') as f:
        cPickle.dump(data, f, -1)

    # Save some images as png for testing
    sub_dir = dir_name + 'png_test_images/'
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    for i in xrange(100):
        img = (255*test_images[i, :, :, 0]).astype(np.uint8)
        img = Image.fromarray(img)
        filename = sub_dir + 'img-{}.png'.format(i)
        img.save(filename)

if __name__ == '__main__':
    download_and_extract()
