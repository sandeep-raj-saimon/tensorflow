import numpy as np

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

NUM_STEPS = 500
MINIBATCH_SIZE = 50
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))

y_true = tf.placeholder(tf.float32,[None,10])
y_pred = tf.matmul(x,W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred,labels = y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Explanation for below code:
#	tf.argmax returns index with largest value across axis of a tensorflow
#	tf.equal accepts two tensors x and y and returns truth value (x==y)
# 	in the below code indices of both y_pred and y_true are compared by tf.equal
correct_mask = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_mask,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for _ in range(NUM_STEPS):
		batch_xs,batch_ys = data.train.next_batch(MINIBATCH_SIZE)
		sess.run(gd_step,feed_dict={x:batch_xs,y_true:batch_ys})
	
	ans = sess.run(accuracy,feed_dict={x:data.test.images,y_true:data.test.labels})
	
print("Accuracy : {:.4%}".format(ans*100))