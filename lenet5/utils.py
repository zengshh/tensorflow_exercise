import os
import gzip
import shutil
import struct
import urllib

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_birth_life_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

def download_one_file(download_url, 
                    local_dest, 
                    expected_byte=None, 
                    unzip_and_remove=False):
    """ 
    Download the file from download_url into local_dest
    if the file doesn't already exists.
    If expected_byte is provided, check if 
    the downloaded file has the same number of bytes.
    If unzip_and_remove is True, unzip the file and remove the zip file
    """
    if os.path.exists(local_dest) or os.path.exists(local_dest[:-3]):
        print('%s already exists' %local_dest)
    else:
        print('Downloading %s' %download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)
        if expected_byte:
            if file_stat.st_size == expected_byte:
                print('Successfully downloaded %s' %local_dest)
                if unzip_and_remove:
                    with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3],'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_dest)
            else:
                print('The downloaded file has unexpected number of bytes')

def download_mnist(path):
    """ 
    Download and unzip the dataset mnist if it's not already downloaded 
    Download from http://yann.lecun.com/exdb/mnist
    """
    safe_mkdir(path)
    url = 'http://yann.lecun.com/exdb/mnist'
    filenames = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
    expected_bytes = [9912422, 28881, 1648877, 4542]

    for filename, byte in zip(filenames, expected_bytes):
        download_url = os.path.join(url, filename)
        local_dest = os.path.join(path, filename)
        download_one_file(download_url, local_dest, byte, True)

def parse_data(path, dataset, flatten):     #parse data from the initial format
    if dataset != 'train' and dataset != 't10k':
        raise NameError('dataset must be train or t10k')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8) #int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1
    
    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols) #uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels

def read_mnist(path, flatten=True, num_train=55000):
    """
    Read in the mnist dataset, given that the data is stored in path
    Return two tuples of numpy arrays
    ((train_imgs, train_labels), (test_imgs, test_labels))
    """
    imgs, labels = parse_data(path, 'train', flatten)
    #print('The shape of images is ', imgs.shape) #=>（60000，784)
    #print('The shape of labels is ', labels.shape) #=>(60000, 10)
    #print('the shape[0] of labels is ', labels.shape[0]) #=> 60000, as it counts from left to right
    
    #in order to shuffle the training set while sustaining the corespondence bewteew imgs and labels
    indices = np.random.permutation(labels.shape[0]) 
    #print('The shape of indices is ', indices.shape)  #=>(60000，）,indices = [2436, 2865, ... , 53687], 0～59999 is distributed randomly
    train_idx, val_idx = indices[:num_train], indices[num_train:]   #pick num_train random index
    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :] #form the new data set based on the random index
    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]

    test = parse_data(path, 't10k', flatten)    #test dataset is independent from the train and eval.
    #print('The type of test is ', type(test)) #=>class tuple. without the arri of test.shape
    
    return (train_img, train_labels), (val_img, val_labels), test

def get_mnist_dataset(batch_size):
    # Step 1: Read in data
    mnist_folder = 'data/mnist'
    download_mnist(mnist_folder)
    train, val, test = read_mnist(mnist_folder, flatten=False) #Attention! donn't need to flatten.

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(10000) # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    eval_data = tf.data.Dataset.from_tensor_slices(val)
    eval_data = eval_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, eval_data, test_data, 
    
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    plt.imshow(image, cmap='gray')
    plt.show()
