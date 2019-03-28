
import tensorflow as tf
import keras
import keras.utils
from keras import utils as np_utils
from keras.datasets import cifar10
import numpy as np
from sklearn import linear_model
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#global variables
num_classes = 10
num_filter = 12
dropout_rate = 0.0
NUM_STEPS = 49500
MINIBATCH_SIZE = 200
l = 40
compression = 0.5
count = 0
IMAGE_SIZE = 32

def add_denseblock(input,num_filter,dropout_rate):
  #print(input.shape)
  global compression
  temp = input
  for _ in range(l):
      with tf.name_scope("dense_block"):
          BatchNorm = tf.layers.batch_normalization(temp)
          relu = tf.nn.relu(BatchNorm)
          conv2D_3_3 = tf.layers.conv2d(inputs=relu,filters=num_filter*compression,kernel_size=[3, 3],padding="same")
          concat = tf.concat([temp,conv2D_3_3],-1)
          temp = concat
          return temp

def add_transition(input,num_filter,dropout_rate):
  with tf.name_scope("transition"):
    global compression
    BatchNorm = tf.layers.batch_normalization(input)
    relu = tf.nn.relu(BatchNorm)
    conv2D_BottleNeck = tf.layers.conv2d(inputs=relu,filters=num_filter*compression,kernel_size=[3, 3],padding="same")
    avg = tf.nn.max_pool(conv2D_BottleNeck,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    return avg

def output_layer(input):
  with tf.name_scope("output_layer"):
    global compression
    BatchNorm = tf.layers.batch_normalization(input)
    relu = tf.nn.relu(BatchNorm)
    print(relu.shape)
    AvgPooling = tf.nn.max_pool(relu,ksize=[1,4,4,1],strides=[1,4,4,1],padding='VALID')
    flat = tf.contrib.layers.flatten(AvgPooling)
    return flat
  
# Load CIFAR10 Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.float32(x_train)
y_train = np.float32(y_train)
y_test = np.float32(y_test)
print(y_test.shape)
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# placeholder
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

# placeholder for string features extracted from our model
features = tf.placeholder(tf.string,[None,output.shape[1]],name="features")

# optimizing the network
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


# image augmentation

from math import pi

def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images)
    
    
    X = tf.placeholder(tf.float32, shape = (None, 32, 32, 3))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate
	
# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
rotated_imgs = rotate_images(x_train[:1000], -90, 90, 15)
x_train = np.concatenate((x_train, rotated_imgs), axis=0)
print("rotated")
print(x_train.shape)
repeat = np.repeat(y_train[:1000], 15)  
repeat = repeat.reshape(15000,1)
y_train = np.concatenate((y_train,repeat),axis=0)
print(y_train.shape)


def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([32, 32], dtype = np.int32)
    
    X_scale_data = []
    X = tf.placeholder(tf.float32, shape = (1,32, 32, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)# on each image 3 operations are perfomred
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data

scaled_imgs = central_scale_images(x_train[1000:3000], [0.90, 0.75, 0.60])
x_train = np.concatenate((x_train, scaled_imgs), axis=0)
print("central scale")
print(x_train.shape)
repeat = np.repeat(y_train[1000:3000], 3)  
repeat = repeat.reshape(6000,1)
y_train = np.concatenate((y_train,repeat),axis=0)
print(x_train.shape)
print(y_train.shape)

# translation of the image
from math import ceil, floor

def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4# each image is translated 4 times
    X_translated_arr = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3), 
				    dtype = np.float32)
            X_translated.fill(1.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
			 w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr
	
translated_imgs = translate_images(x_train[3000:6000])
x_train = np.concatenate((x_train, translated_imgs), axis=0)
print("translate")
print(x_train.shape)
repeat = np.repeat(y_train[3000:6000], 4)  
repeat = repeat.reshape(12000,1)
y_train = np.concatenate((y_train,repeat),axis=0)
print(y_train.shape)

# flipping of images
def flip_images(X_imgs):
    X_flip = []
    
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})# on each image 3 operations are performed
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip
	
flipped_images = flip_images(x_train[6000:10000])
x_train = np.concatenate((x_train, flipped_images), axis=0)
print("flip")
print(x_train.shape)
repeat = np.repeat(y_train[6000:10000], 3)  
repeat = repeat.reshape(12000,1)
y_train = np.concatenate((y_train,repeat),axis=0)

x_train, y_train = shuffle(x_train, y_train)

print(x_train.shape)
print(y_train.shape)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  clf = linear_model.SGDClassifier(loss='log', tol=1e-3)

  saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('./')) # checkpoint file path
  
  for i in range(NUM_STEPS):
    batch_xs,batch_ys = next_batch(MINIBATCH_SIZE)
    #result = sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0}) 
    feature = sess.run(output,feed_dict={x:batch_xs})
    print(i)
    #perform_svm(feature,batch_ys)
  #print(sess.run(fe,feed_dict={x:batch_xs,features:feature}))
  #test(sess)
  
    clf.partial_fit(feature, batch_ys,classes = np.unique(batch_ys))
    
  test_feature = sess.run(output,feed_dict={x:x_test[:10000]})
  clf.predict(test_feature)
  print(test_feature.shape)
  print(clf.score(test_feature,y_test[:10000])*100)
