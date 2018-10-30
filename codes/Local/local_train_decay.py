import tensorflow as tf
import numpy as np
import argparse 
import sys 
import random
import time
import os
from tensorflow.python import debug as tf_debug
import h5py 
from scipy.sparse import load_npz


###### loading data ############


data_train=load_npz('data_train.npz');
data_train=data_train.toarray();

data_test=load_npz('data_test.npz');
data_test=data_test.toarray();

y_train=np.load('y_train.npy');

y_test=np.load('y_test.npy');





############### accuracy prediction function ##################

def accuracy_eval(y_test,y_pred):
    s=0;
    
    y_pred = np.argmax(y_pred,axis=1);
    
    for i in range(len(y_test)):
        if y_test[i,y_pred[i]]>0:
            s+=1;
    return s/np.float(len(y_test));













batch_size=4000;

num_examples = len(data_train);
num_batches_per_epoch = int(num_examples/batch_size);
num_epochs=100;

reg_constant=.001
initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant)



graph=tf.Graph()

with graph.as_default():
    global_step = tf.Variable(0, trainable=False);
    
    starter_learning_rate = 0.004
    
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps=num_examples/batch_size, decay_rate=0.94, staircase=True)
    
    inp =tf.placeholder(shape=(None,data_train.shape[1]),dtype=tf.float64);
    
    labels=tf.placeholder(shape=(None,50),dtype=tf.float64);
    
    W=tf.get_variable('W',shape=(data_train.shape[1],50),dtype=tf.float64,                                                 initializer=initializer, regularizer=regularizer);
    
    b=tf.get_variable('b',shape=(50,),dtype=tf.float64,                     regularizer=regularizer );
    
    out=tf.add(tf.matmul(inp,W),b);
    
    out_logits=tf.nn.softmax(out);
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES);
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=out));
    loss= loss + reg_constant*sum(reg_losses);
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step);
    

    
    
    



############################## trianing of model #########################

cost_plot=[]
test_plot=[]
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run();
    for curr_epoch in range(num_epochs):
        test_accuracy = 0
        train_cost = 0
        start = time.time()



        #iterating over total number of batchs
        for batch in range(num_batches_per_epoch):

            # Getting the index
            
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            
#             data_train=np.zeros((batch_size,len(word_to_index)));
#             for ind in range(len(indexes)):
#                 for j in range(len( x_train[indexes[i]] ) ):
#                     data_train[ind][  x_train[indexes[i]]  ]=1;
            
            
            batch_train_inputs = data_train[indexes];
            batch_train_inputs=batch_train_inputs;
            batch_train_targets=y_train[indexes];
            
            

            # feed dictionay for training
            feed = {inp: batch_train_inputs,
                    labels: batch_train_targets,}

            batch_cost, _ = session.run([loss, optimizer], feed)
            train_cost += batch_cost*batch_size




#         # Shuffle the data
#         shuffled_indexes = np.random.permutation(num_examples)## shuffling data for next epoch
#         data_train = data_train[shuffled_indexes];
#         data_train = data_train[shuffled_indexes];

        # Metrics mean
        train_cost /= num_examples;
        
        cost_plot.append(train_cost);

        if(curr_epoch%1==0):
            
            test_start=time.time();
            
            batch_test_inputs = data_test;
            batch_test_inputs=batch_test_inputs;
            
            
            # feed dictionary for test data to calculate test CER
            
            feed_test = {
                        inp: batch_test_inputs,
                        
                        }
            
            y_pred = session.run(out_logits,feed_test)

            test_accuracy=accuracy_eval(y_test,y_pred);
            test_plot.append(test_accuracy);
            
            test_end=time.time();
            
            
            log = "Epoch {}/{}, train_cost = {:.3f}, test_accuracy = {:.3f}, time = {:.3f},test-time={:.3f}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, test_accuracy, time.time() - start,test_end-test_start))

    
    
np.save('cost_plot_decay',cost_plot)
np.save('test_plot_decay',test_plot)    
    
    


# In[39]:

