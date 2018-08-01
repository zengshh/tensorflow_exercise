import tensorflow as tf
import os 
import matplotlib.pyplot as plt

MODEL_PATH = "../checkpoints"
MODEL_NAME = "lenet.ckpt"

mnist_vars = [['conv1/weights', 'conv1/biases'],
                ['conv2/weights', 'conv2/biases'],
                ['fc/weights', 'fc/biases'],
                ['logits/weights', 'logits/biases']]

def draw_distribution(data):
    plt.hist(data, 100)
    plt.xlabel('Vars')
    plt.ylabel('Frequency')
    plt.title('Vars Distribution')
    plt.show()
    
def dump_data(arr, element):
    """
    arr: input data, should be (n, ) array or sequence of (n, ) arrays
    elemet: data name, may contain / .
    """
    #method1: only search in the leftest dime of array.  
    #for index, value in enumerate(arr):
    #    print('{0} : {1} \n'.format(index, value))

    #method2: search every element from right to left dimentionally.
    str_list = element.split('/')
    print(str_list)

    if str_list[-1] == 'biases':
        name = element.replace('/', '_')    #refer to python string for more operations
        with open(name+'.txt', 'w') as data:
            L = arr.size
            for l in range(L):
                print(arr[l], end=' ', file=data)
    elif str_list[-1] != 'Adam_1' and str_list[-1] != 'Adam':
        if element[:4] == 'conv' : #here assume we have know the name
            name = element.replace('/', '_')    #refer to python string for more operations
            with open(name+'.txt', 'w') as data:
                N, H, W, C = arr.shape
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                print(arr[n, h, w, c], end=' ', file=data)
                            print(end='\n', file = data)
                        print(end='\n', file = data)
                    print(end='\n', file = data)

        elif element[:2] == 'fc':
            name = element.replace('/', '_')
            with open(name+'.txt', 'w') as data:
                H, W = arr.shape
                for h in range(H):
                    for w in range(W):
                        print(arr[h, w], end=' ', file=data)
                    print(end='\n', file=data)
        
        elif element[:5] == 'logit':       #the last layer, usually it is fc
            name = element.replace('/', '_')
            with open(name+'.txt', 'w') as data:
                H, W = arr.shape
                for h in range(H):
                    for w in range(W):
                        print(arr[h, w], end=' ', file=data)
                    print(end='\n', file=data)

#-------------------------------------------------------------------------------------#
with tf.Session() as sess:
    """
    1. to trace all the variable in the former model. 
    2. tf.python.pywrap_tensorflow.NewCheckpointReader() is equal to the below.
    3. MODEL_NAME is not a file, but connect with several file on the MODEL_PATH folder
    """
    reader = tf.train.NewCheckpointReader(os.path.join(MODEL_PATH, MODEL_NAME))
    vars = reader.get_variable_to_shape_map()
    #print(vars)    #display all the variable name and its shape
    
    for element in vars:
        arr = reader.get_tensor(element)    # type numpy.ndarray
        print("tensor name : ", element, end='\t')
        print(arr.shape)

        dump_data(arr, element)
        #arr = arr.reshape(1, arr.size)  #it costs much time and report warning.
        arr = arr.reshape(-1) #equal to reshape(arr.size), but this is better.  shape is (n, )
        draw_distribution(arr)
        
