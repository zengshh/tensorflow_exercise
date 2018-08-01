import tensorflow as tf
import os 

MODEL_PATH = "checkpoints"
MODEL_NAME = "lenet.ckpt"

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
        arr = reader.get_tensor(element)
        print("tensor name : ", element, end='\t')
        print(arr.shape)
        
        #method1: only search in the leftest dime of array.  
        #for index, value in enumerate(arr):
        #    print('{0} : {1} \n'.format(index, value))

"""
        #method2: search every element from right to left dimentionally.
        if element == 'layers/weights' : #here assume we have know the name
            with open('weights.txt', 'w') as data:
                N, H, W, C = arr.shape
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                print(arr[n, h, w, c], end=' ', file=data)
                            print(end='\n', file = data)
                        print(end='\n', file = data)
                    print(end='\n', file = data)
"""            

