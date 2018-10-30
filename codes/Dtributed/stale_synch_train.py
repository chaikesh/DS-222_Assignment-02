
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import argparse 
from tensorflow.python import debug as tf_debug
import h5py 
import sys 
import random
import time
import os
from scipy.sparse import load_npz

data_train=load_npz('data_train.npz');
data_train=data_train.toarray();

data_test=load_npz('data_test.npz');
data_test=data_test.toarray();

y_train=np.load('y_train.npz');
y_test=np.load('y_test.npz');

def accuracy_eval(y_test,y_pred):
    s=0;
    
    y_pred = np.argmax(y_pred,axis=1);
    
    for i in range(len(y_test)):
        if y_test[i,y_pred[i]]>0:
            s+=1;
    return s/len(y_test);

stale_parameter=int(sys.argv[1][-2:]);
print('staleness factor chosen is {}'.format(stale_parameter));

parameter_servers=["10.1.1.254:2225"];
workers=["10.1.1.253:2223","10.1.1.252:2224"];
cluster = tf.train.ClusterSpec({"ps":parameter_servers,"worker":workers});

tf.app.flags.DEFINE_string("job_name","","'ps' / 'worker'");
tf.app.flags.DEFINE_integer("task_index",0,"Index of task within the job");
FLAGS=tf.app.flags.FLAGS;

config=tf.ConfigProto();
config.gpu_options.allow_growth=True;
config.allow_soft_placement=True;
config.log_device_placement=True;

server=tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index,config=config)

if FLAGS.job_name=='ps':
    server.join();
elif FLAGS.job_name=='worker':
    
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):

        batch_size=4000;
        num_examples = len(data_train);
        num_batches_per_epoch = int(num_examples/batch_size);
        num_epochs=100;
        reg_constant=.001
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant)

        global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False,dtype=tf.int32)
    
        starter_learning_rate = 0.004
        inp =tf.placeholder(shape=(None,data_train.shape[1]),dtype=tf.float64);
        labels=tf.placeholder(shape=(None,50),dtype=tf.float64);
        W=tf.get_variable('W',shape=(data_train.shape[1],50),dtype=tf.float64, initializer=initializer, regularizer=regularizer)
        b=tf.get_variable('b',shape=(50,),dtype=tf.float64, regularizer=regularizer )
        out=tf.add(tf.matmul(inp,W),b);
        out_logits=tf.nn.softmax(out);
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES);
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=out));
        loss= loss + reg_constant*sum(reg_losses);
        optimizer=tf.train.AdamOptimizer(learning_rate=starter_learning_rate);        
        replicated_opt=tf.contrib.opt.DropStaleGradientOptimizer(opt=optimizer,staleness=stale_parameter);        
        final_optimizer = replicated_opt.minimize(loss, global_step=global_step);

        init_op = tf.global_variables_initializer()
        
        tf.summary.scalar("loss", loss)   
        summary_op = tf.summary.merge_all();
        sv_obj = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),                            global_step=global_step,                            init_op=init_op);
               
    with sv_obj.prepare_or_wait_for_session(server.target) as session:

        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph());
        cost_plot_task0=[];
        cost_plot_task1=[];
        for curr_epoch in range(num_epochs):
            test_accuracy = 0
            train_cost = 0
            start = time.time()
            for batch in range(num_batches_per_epoch):
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
                batch_train_inputs = data_train[indexes];
                batch_train_inputs=batch_train_inputs;
                batch_train_targets=y_train[indexes];

                feed = {inp: batch_train_inputs,
                        labels: batch_train_targets,}

                batch_cost, _ = session.run([loss, final_optimizer], feed)
                train_cost += batch_cost*batch_size

            train_cost /= num_examples;

            if FLAGS.task_index==0:
                cost_plot_task0.append(train_cost);
            else:
                cost_plot_task1.append(train_cost);

            if(curr_epoch%1==0):

                test_start=time.time();

                batch_test_inputs = data_test;
                batch_test_inputs=batch_test_inputs;
                feed_test = {
                            inp: batch_test_inputs,

                            }

                y_pred = session.run(out_logits,feed_test)
                test_accuracy=accuracy_eval(y_test,y_pred);
                test_end=time.time();
                print(log.format(curr_epoch+1, num_epochs, train_cost, test_accuracy, time.time() - start,test_end-test_start))
                
    sv_obj.stop();

