import tensorflow as tf
import numpy as np

x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
	x = tf.placeholder(tf.float32,shape=(5,10))
	w = tf.placeholder(tf.float32,shape=(10,1))
	b = tf.fill((5,1),-1.)
	xw = tf.matmul(x,w)
	
	xbw = xw + b
	s = tf.reduce_max(xbw)
	
	with tf.Session() as sess:
		out = sess.run(s,feed_dict={x:x_data,w:w_data})
		