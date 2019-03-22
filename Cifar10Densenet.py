
import tensorflow as tf
import keras
import keras.utils
from keras import utils as np_utils
from keras.datasets import cifar10
import numpy as np


#global variables
num_classes = 10
num_filter = 12
dropout_rate = 0.0
NUM_STEPS = 100
MINIBATCH_SIZE = 300
l = 40
compression = 0.5
count = 0

def add_denseblock(input,num_filter,dropout_rate):
  #print(input.shape)
  global compression
  temp = input
  for _ in range(l):
      with tf.name_scope("dense_block"):
          BatchNorm = tf.layers.batch_normalization(temp)
          relu = tf.nn.relu(BatchNorm)
         # print("Problem")
         # print(temp.shape)
          conv2D_3_3 = tf.layers.conv2d(inputs=relu,filters=num_filter*compression,kernel_size=[3, 3],padding="same")
         # print(conv2D_3_3.shape)
          concat = tf.concat([temp,conv2D_3_3],-1)
         # print(concat.shape)
          temp = concat
         # print(temp.shape)
          return temp

def add_transition(input,num_filter,dropout_rate):
  with tf.name_scope("transition"):
    global compression
    BatchNorm = tf.layers.batch_normalization(input)
    relu = tf.nn.relu(BatchNorm)
    conv2D_BottleNeck = tf.layers.conv2d(inputs=relu,filters=num_filter*compression,kernel_size=[3, 3],padding="same")
    avg = tf.nn.avg_pool(conv2D_BottleNeck,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    return avg

def output_layer(input):
  with tf.name_scope("output_layer"):
    global compression
    BatchNorm = tf.layers.batch_normalization(input)
    relu = tf.nn.relu(BatchNorm)
    AvgPooling = tf.nn.avg_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    flat = tf.contrib.layers.flatten(AvgPooling)
    #rint(flat.shape)
    #utput = full_layer(full1_drop,10)

    output = tf.layers.dense(flat,10)
    return output
# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.float32(x_train)
y_train = np.float32(y_train)
x_test = np.float32(x_test)
y_test = np.float32(y_test)

img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# convert to one hot encoing 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#data = tf.keras.layers.Input(shape=(img_height, img_width, channel,))
x = tf.placeholder(tf.float32,[None,32,32,3],name="x")
y_= tf.placeholder(tf.float32,[None,10],name="y_")
keep_prob = tf.placeholder(tf.float32,name="keep_prob")

conv1 = tf.layers.conv2d(inputs=x,filters=12,kernel_size=[3, 3],padding="same", activation=None)

#architecture of denseNet starts now
First_Block = add_denseblock(conv1, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels = y_))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(output ,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_mask,tf.float32))

def next_batch(num):
  global count
  x,y = x_train[count:count+num],y_train[count:count+num]
  count = (count+num)%len(x_train)
  return x,y

def test(sess):
  X = x_test.reshape(10,1000,32,32,3)
  Y = y_test.reshape(10,1000,10)
  acc = np.mean([sess.run(accuracy,feed_dict={x:X[i],y_:Y[i],keep_prob:1.0})for i in range(10)])
  print("Accuracy : {:.4}%".format(acc*100))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for i in range(NUM_STEPS):
    batch_xs,batch_ys = next_batch(MINIBATCH_SIZE)
    #sess.run(conv1,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0})
    sess.run(cross_entropy,feed_dict={x:batch_xs,y_:batch_ys})
  #    sess.run(train_step,feed_dict={x:batch_xs.eval(),y_:batch_ys.eval(),keep_prob:1.0})  
  test(sess)
