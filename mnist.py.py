import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

# global data
count = 0
BATCH_SIZE = 125
NUM_STEPS = 1500

#load mnist data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# convert to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# placeholder for input data
x = tf.placeholder(tf.float32,shape=[None,784])
y_actual = tf.placeholder(tf.float32,shape=[None,10])

# weights 
w = tf.Variable(tf.zeros([784,10]))

# predicted value
y_pred = tf.matmul(x,w)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred,labels = y_actual))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Explanation for below code:
#	tf.argmax returns index with largest value across axis of a tensorflow
#	tf.equal accepts two tensors x and y and returns truth value (x==y)
# 	in the below code indices of both y_pred and y_true are compared by tf.equal
correct_mask = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_mask,tf.float32))

# generating next batch for training
def next_batch(num):
  global count
  x,y = x_train[count:count+num],y_train[count:count+num]
  count = (count+num)%len(x_train)
  return x,y

with tf.Session() as sess:
  
  sess.run(tf.global_variables_initializer())
  
  for _ in range(NUM_STEPS):
    
    batch_xs,batch_ys = next_batch(BATCH_SIZE)
    
    batch_xs = batch_xs.reshape(BATCH_SIZE,784)
    batch_ys = batch_ys.reshape(BATCH_SIZE,10)
    
    sess.run(gd_step,feed_dict={x:batch_xs,y_actual:batch_ys})
  
  x_test = x_test.reshape(10000,784)
  y_test = y_test.reshape(10000,10)
  
  ans = sess.run(accuracy,feed_dict={x:x_test,y_actual:y_test})
	
print("Accuracy : {:.2%}".format(ans))


