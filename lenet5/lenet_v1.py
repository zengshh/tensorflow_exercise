""" 
Author: zengshh
Email: zengshh2016@outlook.com
Last change date: 7/30/2018
More detail: 
  1.Using convolutional net on MNIST dataset of handwritten digits
  2.MNIST dataset: http://yann.lecun.com/exdb/mnist/
  3.This program reveal the whole procedure of building\training\validating one conv-net--mnist here.
  4.Class mode provides a clear and reuseable way to achieve it, though it seems a little heavy. 
  5.Using L2 regularization and learning-rate exponential decay to optimize, which helps convergencing quicker and improves accuroacy .
  6.Provide a way to abserve the distribution of inter-layer results. 
"""

import os
import time 
import tensorflow as tf
import utils
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def conv_relu(inputs, filters, k_size, stride, padding, scope_name = 'conv', regular = None):
    '''
    A method that does convolution + relu on inputs
    '''
    #with the scope, initializing several conv_relu will produce different variables, even they seem to share the same name
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:   
        in_channels = inputs.shape[-1]      #get the C of input:(N, H, W, C)
        weights = tf.get_variable('weights',
                                [k_size, k_size, in_channels, filters], #(K, K, C, N)
                                initializer=tf.truncated_normal_initializer() )
        biases = tf.get_variable('biases',
                                [filters], # equal to [filters, ], produce a 1-dim vector
                                initializer=tf.random_normal_initializer() )
        conv = tf.nn.conv2d(inputs, weights, strides = [1, stride, stride, 1], padding = padding)
        if regular != None:     #L2 regularization
            tf.add_to_collection('l2_losses', regular(weights))
    
    return tf.nn.relu(conv + biases, name = scope.name) # + uses broadcast method.

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:   
        pool = tf.nn.max_pool(inputs,
                            ksize = [1, ksize, ksize, 1],
                            strides = [1, stride, stride, 1],
                            padding = padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc', regular = None):
    '''
    A fully connected linear layer on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]       #get the last dime of inputs 
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
        
        if regular != None:     #L2 regularization
            tf.add_to_collection('l2_losses', regular(w))
    
    return out

#the methods of class don't return values

class ConvNet(object):
    def __init__(self):
        #self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)      #used for drop-out
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_train= 55000     #train dataset size.
        self.n_test = 10000     #test dataset size.
        self.n_eval = 5000      #eval dataset size.
        self.training = True
        self.dump = False

        self.lr_base = 0.002  #when value is 0.1, the optimizer cannot convergence.
        self.lr_decay = 0.95
        self.regularization_rate = 0.0001

    def get_data(self):
        with tf.name_scope('data'):
            #1.construct the dataset for train\validate\test.
            train_data, eval_data, test_data = utils.get_mnist_dataset(self.batch_size) # without val_data
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                               train_data.output_shapes)
            #print(train_data.output_shapes)    #((None, 28, 28), (None, 10))
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.eval_init = iterator.make_initializer(eval_data)    
            self.test_init = iterator.make_initializer(test_data)    
            
            #2.construct the dataset only for dumping the inter-layers 
            test = utils.parse_data('data/mnist', 't10k', False)   #refer to parse_data function
            test_set = tf.data.Dataset.from_tensor_slices(test)    #construct the dataset
            iterator1 = test_set.make_one_shot_iterator()

            if self.dump == False:
                img, self.label = iterator.get_next()               #fetch batch-size samples, but not always equals to batch_size.
                #print(img.shape, self.label.shape)             #(?, 28, 28)  (?, 10)
            else:    
                img, self.label = iterator1.get_next()
                #print(img.shape, self.label.shape)             #(28, 28)  (10, )
                length = self.label.shape[0]                    #equals to label.size
                self.label = tf.reshape(self.label, shape=[1,length])  #define the shaps as (1, 10) forcedly

            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])   # -1 represents not-specific, here should be 1
            # reshape the image to make it work with tf.nn.conv2d

    def inference(self):
        '''
        Build the model according to the description we've shown in class
        '''
        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)

        self.conv1 = conv_relu(inputs = self.img,
                            filters = 32,
                            k_size = 5,
                            stride = 1,
                            padding = 'SAME',
                            scope_name = 'conv1',
                            regular = regularizer)
        pool1 = maxpool(self.conv1, 2, 2, 'VALID', 'pool1')  # Pay attention to the position
        
        self.conv2 = conv_relu(inputs = pool1,
                            filters = 64,
                            k_size = 5,
                            stride = 1,
                            padding = 'SAME',
                            scope_name = 'conv2',
                            regular = regularizer)
        pool2 = maxpool(self.conv2, 2, 2, 'VALID', 'pool2')  # Pay attention to the position

        feature_dim = pool2.shape[1]*pool2.shape[2]*pool2.shape[3] # H*W*C
        pool2 = tf.reshape(pool2, [-1, feature_dim])     #flatten the NHWC into two dime. -1 should be pool2.shape[0]

        self.fc = fully_connected(pool2, 1024, 'fc', regularizer)
        dropout = tf.nn.dropout(x = tf.nn.relu(self.fc), keep_prob = self.keep_prob, name = 'relu_dropout')

        self.logits = fully_connected(dropout, self.n_classes, 'logits', regularizer)

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        tf.nn.softmax_cross_entropy_with_logits
        softmax is applied internally
        don't forget to compute mean cross all sample in a batch
        '''
        with tf.name_scope('loss'):
            #return 1-D tensor of length batch-size of the same type as logits
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss') + tf.add_n(tf.get_collection('l2_losses'))

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        Don't forget to use global step
        '''
        self.learning_rate = tf.train.exponential_decay(
                            self.lr_base,
                            self.gstep,
                            self.n_train/self.batch_size,
                            self.lr_decay,
                            staircase = True)

        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,  global_step=self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        Remember to track both training loss and test accuracy
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()
 
    def evaluate(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))   #wrong? should be reduce_mean?See later

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.summary()  #used for tensorboard for vision

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                #if (step + 1) % self.skip_step == 0:
                #    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        
        saver.save(sess, 'checkpoints/lenet.ckpt')
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        
        return step             #attentin! step was returned here!! and g_step only changed while training.

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_eval))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and testing
        '''
        writer = tf.summary.FileWriter('./graphs/', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            #the restore operation is not necessary here while training.
            #ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_starter/checkpoint'))
            #if ckpt and ckpt.model_checkpoint_path:
            #   saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval() #actually, the step keeps the same change-law as gstep.
            print("the initial value of step is %d" % step)

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.eval_init, writer, epoch, step)   #here step is not changed.
                print("current learning-rate is %f \n" % self.learning_rate.eval())
                print("current step is %d \n" % step)
        writer.close()
    
    def test(self):
        start_time = time.time()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
               saver.restore(sess, ckpt.model_checkpoint_path)

            if self.dump == False:      
                sess.run(self.test_init)    #initilize the dataset with test_data. Then compute-graph keeps unchanged. similar to test_once
                self.training = False
                total_correct_preds = 0
                try:
                    while True:
                        accuracy_batch = sess.run(self.accuracy)
                        total_correct_preds += accuracy_batch
                except tf.errors.OutOfRangeError:
                    pass

                print('Accuracy at test dataset : {0} '.format(total_correct_preds/self.n_test))
                print('Took: {0} seconds'.format(time.time() - start_time))

        #-------------------------------------------------------------#
        #       dump the output of inter-layers                       # 
        #-------------------------------------------------------------#
            else:
                test_num = 100        #numer of test, for analysing hte inter-layers output.
                test_list = []                                      #define an empty list
                total_correct_preds = 0 
                for i in range(test_num):
                    accuracy_batch = sess.run(self.accuracy)
                    total_correct_preds += accuracy_batch
                    test_list.append(self.conv2.eval())             #here collect the conv2 output. you can change it
                
                test_array = np.array(test_list)                    #if conv-layers, shape is (test_num, h, w, c)
                test_array = test_array.reshape((-1, ))                 #convert to 1-D array
                print('Average accuracy at {0} tests : {1} '.format(test_num, total_correct_preds/test_num))
                print('Max: %f   Min: %f  Mid: %f' %(np.max(test_array), np.min(test_array), np.median(test_array) ))
                plt.hist(test_array, 100)
                plt.show()

if __name__ == '__main__':
    model = ConvNet()
    model.dump = True              #if you want to observe the inter-layers result distribution, please uncomment it. It should be commented while training 
    model.build()
    #model.train(n_epochs=40)       #after trained, this is not necessary for test and it saves time.
    model.test()
