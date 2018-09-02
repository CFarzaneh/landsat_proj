import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import copy
import os

# Training and Validation data extraction
def data_extraction():

    while True:

        path = input('If training and validation files are not present in the current directory, then please enter the absolute directory path: ')
        
        if  os.path.exists(path) or path == "":
            Train_X,Train_Y = data_file(path)
            Val_X,Val_Y = data_file(path,train = False)
            return Train_X, Train_Y, Val_X, Val_Y


def data_file(path, train = True):
    if train:
        while True:
            train_file = input('Please enter correct training data filename: ')
            train_label = input('Please enter correct training data label filename: ')

            if os.path.exists(path+train_file) and os.path.exists(path +train_label):
                print ("Loading Training Data *******************************")
                Train_X = np.load(path + train_file)
                Train_Y = np.load(path + train_label)
                if len(Train_X) == len(Train_Y):
                    return Train_X, Train_Y
                else:
                    print ("Data and label files have different length")
                    continue

            else:
                print ("No such files exist, Please check the names again")


    if not train:
        while True:
            val_file = input('Please enter correct validation data filename: ')
            val_label = input('Please enter correct validation data label filename: ')

            if os.path.exists(path+val_file) and os.path.exists(path +val_label):
                print ("Loading Validation Data ******************************")
                Val_X = np.load(path + val_file)
                Val_Y = np.load(path + val_label)
                if len(Val_X) == len(Val_Y):
                    return Val_X, Val_Y
                else:
                    print ("Data and label files have different length")
                    continue

            else:
                print ("No such files exist, Please check the names again")






# LSTM function to get the logits (predictions) 
def LSTM(DataX, drop_prob, weights, biases):
 
    # Reshaping to (batch_size*n_steps, n_input)
    #x1 == (256 batch *23 steps, 8 inputs)
    #x2 == (256 batch *23 steps, 72 inputs)
    DataX = tf.reshape(DataX, [-1, n_input])
    

    DataX_in = tf.matmul(DataX, weights['in']) + biases['in']
    
    # Reshape back for LSTM 
    DataX_in = tf.reshape(DataX_in, [-1, n_steps, n_hidden])


    # Define a lstm cell 
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias = 1.0, state_is_tuple = True)

    ''' 
    Drop out Wrapper
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob= drop_prob, output_keep_prob= drop_prob)
    '''

    # Setting the initial state
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # LSTM cell outputs/states
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, DataX_in, initial_state = _init_state, time_major= False, dtype=tf.float32)




    # unstack LSTM cell outputs 
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))



    # return the output value at the last (23rd) time step. 
    return tf.matmul(outputs[-1], weights['out']) + biases['out']






# Function to extract mini-batches of data for training 
def batch_data(x_data,y_data,batch_size):

  global counter
  image_batch = x_data[counter : counter + batch_size]
  label_batch = y_data[counter : counter + batch_size]
  

  if len(image_batch) == batch_size:
    counter = counter + batch_size
    return image_batch , label_batch 

  elif len(image_batch) != 0 and  len(image_batch) != batch_size:
    add_patch = batch_size - len(image_batch)
    add_data = x_data[0 : add_patch]
    add_label = y_data[0 : add_patch]
    image_batch = np.append(image_batch,add_data, axis=0)
    label_batch = np.append(label_batch,add_label, axis =0)
    counter = add_patch
    return image_batch , label_batch

   
         
  else:
    counter = 0
    image_batch = x_data[counter : counter + batch_size]
    label_batch = y_data[counter : counter + batch_size]
    counter = counter + batch_size
    return image_batch , label_batch 


   

# Training and Validation data files are stored as numpy file

Training_X ,Training_Y, Validation_X, Validation_Y = data_extraction()


# Global counter for batch extraction
counter = 0

# Parameters
learning_rate = 0.0001
display_step = 100
test_step = 2500
batch_size = 256
num_epochs = 100
total_training_samples = len(Training_Y)
epoch_iters = int(total_training_samples/batch_size)
training_iters = epoch_iters*num_epochs
lambda_l2_reg = 0.005



# Network Parameters
n_input = 72 
n_steps = 23 # timesteps
n_hidden = 1024# hidden layer num of features
n_classes = 8 #total classes 


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'in': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}




# Logits (predictions) values
prediction = LSTM(x, keep_prob, weights, biases)


# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
'''# use regularization after setting other parameters 
l2 = lambda_l2_reg * sum(tf.nn.l2_loss(var) for var in tf.trainable_variables() if not  "Bias" in var.name)
cost += l2'''

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluation
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initializing the variables and 
init = tf.global_variables_initializer()




# Graph
with tf.Session() as sess:

    sess.run(init)
    
    # Saver class object to save parameters at different checkpoints
    saver = tf.train.Saver(max_to_keep=10000)

    # Directory to save checkpoints
    saver_path = input("Please enter checkpoints-saver directory name: ")

    if saver_path[-1] != "/":
        saver_path += "/"
        
    if not os.path.exists(saver_path):
        os.makedirs(saver_path)

    step = 1
    
    while step < training_iters:

        # Training batch
        batch_x, batch_y = batch_data(Training_X,Training_Y,batch_size)

        # Running optimization / Backprop 
        # Set drop_out probability to 0.5 after checking and setting other parameters, initially set it to 1.0 
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        if step % display_step == 0:

            # Batch accuracy
            acc_t = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

            # Batch loss
            loss_t = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

            print ("Step " + str(step) + ", Batch Loss= " + "{:.6f}".format(loss_t) + ", Batch Accuracy= " + "{:.5f}".format(acc_t))





        if step % test_step == 0:
            
            # Save learnable parameters
            saver.save(sess, saver_path , global_step=step)
 
            print ("Validation Accuracy:")

            # Calculate validation accuracy and loss
            acc_v,loss_v = (sess.run([accuracy,cost], feed_dict={x: Validation_X, y: Validation_Y, keep_prob: 1.0}))
            

            print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            print ("Step " + str(step) + ", Validation Loss= " + "{:.6f}".format(loss_v) + ", Validation Accuracy= " + "{:.5f}".format(acc_v))

            print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            

           

        step += 1

   




    print ("Optimization Done!")