import tensorflow as tf
import os 

MODEL_PATH = "model"
GRAPH_NAME = "model.ckpt.meta"
#DATA_NAME = "model.ckpt.data-00000-of-00001"
MODEL_NAME = "model.ckpt"

#reload the graph from the .meta. or we need to define the 
#variables, strictly as the former does.
saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, GRAPH_NAME))
graph = tf.get_default_graph()

with tf.Session() as sess:
    #check if there exists a checkpoint and then restore it
    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    #get the tensor by the exact name. pay attention to :0
    w1 = sess.run(graph.get_tensor_by_name("layers/weights:0"))
    #print(w1)

    """
    1. to trace all the variable in the former model. 
    2. tf.python.pywrap_tensorflow.NewCheckpointReader() is equal to the below.
    3. MODEL_NAME is not a file, but connect with several file on the MODEL_PATH folder
    """
    reader = tf.train.NewCheckpointReader(os.path.join(MODEL_PATH, MODEL_NAME))
    vars = reader.get_variable_to_shape_map()
    #print(vars)    #display all the variable name and its shape
    
    for element in vars:
        print("tensor name : ", element)
        arr = reader.get_tensor(element)

        #method1: only search in the leftest dime of array.  
        for index, value in enumerate(arr):
            print('{0} : {1} \n'.format(index, value))

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
            

