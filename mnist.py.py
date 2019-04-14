import numpy as np

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

NUM_STEPS = 1000
MINIBATCH_SIZE = 100
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("MNIST_data/", one_hot=True)
LOG_DIR = 'E:\Learn_Data_Science_in_3_Months-master\Tensorflow'

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))

y_true = tf.placeholder(tf.float32,[None,10])
y_pred = tf.matmul(x,W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred,labels = y_true))
tf.summary.scalar('cross_entropy',cross_entropy)
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Explanation for below code:
#	tf.argmax returns index with largest value across axis of a tensorflow
#	tf.equal accepts two tensors x and y and returns truth value (x==y)
# 	in the below code indices of both y_pred and y_true are compared by tf.equal
correct_mask = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_mask,tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:

	train_writer = tf.summary.FileWriter(LOG_DIR+'/train',graph = tf.get_default_graph())
	test_writer = tf.summary.FileWriter(LOG_DIR+'/test',graph = tf.get_default_graph())
	
	sess.run(tf.global_variables_initializer())
	
	for i in range(NUM_STEPS):
		batch_xs,batch_ys = data.train.next_batch(MINIBATCH_SIZE)
		
		summary,_ = sess.run([merged,gd_step],feed_dict={x:batch_xs,y_true:batch_ys})
		train_writer.add_summary(summary,i)
		sess.run(gd_step,feed_dict={x:batch_xs,y_true:batch_ys})
		
		if i %100 == 0:
			acc,loss = sess.run([accuracy,cross_entropy],feed_dict={x:batch_xs,y_true:batch_ys})
			
			print("Iter : " + str(i) +" loss : "+ str(loss) + " training accuracy : " + str(acc))
	
	ans = sess.run(accuracy,feed_dict={x:data.test.images,y_true:data.test.labels})
	
	
print("Accuracy : {:.2%}".format(ans))
