import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from helper import create_training_data, save_leaf, create_training_data_example
import time
import math
from random import shuffle


class Autoencoder(object):
	def __init__(self,learning_rate,batch_size,pointcloud_dim,epochs):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.pointcloud_dim = pointcloud_dim
		self.epoch = epochs
		self.build_network()
	
	
	
	
	
	
	
	
	def encoder(self,y):
		with tf.variable_scope("encoder") as scope:	
							
			#y = tf.reshape(y, [-1, 1024,3])
			#y = tf.contrib.layers.conv1d(y,num_outputs =32,kernel_size=1,stride=1,padding="same",activation_fn=tf.nn.leaky_relu)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			#y = tf.contrib.layers.conv1d(y,num_outputs =64,kernel_size=1,stride=2,padding="same",activation_fn=tf.nn.leaky_relu)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			#y = tf.contrib.layers.conv1d(y,num_outputs =128,kernel_size=1,stride=2,padding="same",activation_fn=tf.nn.leaky_relu)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			#y = tf.contrib.layers.conv1d(y,num_outputs =256,kernel_size=1,stride=1,padding="same",activation_fn=tf.nn.leaky_relu)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			#y = tf.contrib.layers.conv1d(y,num_outputs =128,kernel_size=1,stride=1,padding="same",activation_fn=tf.nn.leaky_relu)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=1e-5,center=True,scale=True)			
			#y = tf.layers.dense(y, 1, activation=tf.nn.leaky_relu)
			
			y = tf.reshape(y, [-1, 3072])

			y = tf.layers.dense(y, 3072, activation=None)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)			
			y = tf.layers.dense(y, 1024, activation=None)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)
			y = tf.layers.dense(y, 512, activation=None)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)	
			y = tf.layers.dense(y, 128, activation=None)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			#y = tf.nn.leaky_relu(y,alpha = 0.2)
			#y = tf.layers.dense(y, 64, activation=None)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			#y = tf.nn.leaky_relu(y,alpha = 0.2)	
			#y = tf.layers.dense(y, 32, activation=None)
			#y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			#y = tf.nn.leaky_relu(y,alpha = 0.2)
			#y = tf.layers.dense(y, 1, activation=None)
						
				
		return y
			
		
		
	def decode(self,x):
		with tf.variable_scope("decoder") as scope:
			x = tf.layers.dense(x, 128, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 256, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 512, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 1024, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 1536, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 2527, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 3072, activation=None)
			#x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=1e-5,center=True,scale=True)
			#x = tf.reshape(x, [-1, 512,3])
			x = tf.reshape(x, [-1, 1024,3])		
		return x
		
	def build_network(self):
		
		self.input = tf.placeholder(tf.float32, [None,self.pointcloud_dim,3], name="real_pointcloud_data")
		self.test = self.encoder(self.input)		
		self.decoder_output = self.decode(self.test)
		#self.decoder_output = tf.reshape(self.decoder_output, [-1, self.pointcloud_dim,3])
		
		self.true = self.input
		self.pred = self.decoder_output
		
		self.loss = tf.reduce_mean(self.true - self.pred)
		
	def train(self):
		print("start training")
		self.train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.training_data = create_training_data_example()
			k = (len(self.training_data) // self.batch_size)
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]
			print("Lengh of the training_data:")
			print(len(self.training_data))
			
			for e in range(1,self.epoch):
				epoch_loss_a = 0.
				
				
				print("data shuffeld")
				np.random.shuffle(self.training_data)
				for i in range(0,k):
					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					
					_, auto_loss = sess.run([self.train_op,self.loss], feed_dict={self.input: self.batch})
					epoch_loss_a += auto_loss
				
				epoch_loss_a /=k
				print("LOSS OF autoencoder: %f" % epoch_loss_a)
					
				
			