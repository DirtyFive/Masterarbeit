import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from helper import create_training_data, save_leaf, create_training_data_example
import time
import math
from random import shuffle
from pyntcloud import PyntCloud
import trimesh
import pandas as pd
from plyfile import PlyData, PlyElement
import random

class PCGAN(object):

	def __init__(self,is_training,epoch,pointcloud_dim,checkpoint_dir, learning_rate,z_dim,batch_size,beta1,beta2):		
		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.z_dim = z_dim
		self.pointcloud_dim = pointcloud_dim
		self.training = is_training
		self.batch_size = batch_size
		self.epoch = epoch
		self.checkpoint_dir = checkpoint_dir
		self.leaf_fake = "leaf_fake"
		self.leaf_real = "leaf_real"
		self.build_network()
		
	def generator(self,z):
		with tf.variable_scope("generator") as scope:
			x = tf.layers.dense(z, 128, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 256, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 512, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 1024, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 1536, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 2527, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			x = tf.nn.leaky_relu(x,alpha = 0.2)
			x = tf.layers.dense(x, 3072, activation=None)
			x = tf.layers.batch_normalization(x,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			#x = tf.reshape(x, [-1, 512,3])
			#x = tf.reshape(x, [-1, 1536])			
		return x
		
	def discriminator(self,y,reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()			
			y = tf.layers.dense(y, 3072, activation=None)
			y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)			
			y = tf.layers.dense(y, 1024, activation=None)
			y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)
			y = tf.layers.dense(y, 512, activation=None)
			y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)	
			y = tf.layers.dense(y, 128, activation=None)
			y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)
			y = tf.layers.dense(y, 64, activation=None)
			y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)	
			y = tf.layers.dense(y, 32, activation=None)
			y = tf.layers.batch_normalization(y,momentum=0.9,epsilon=self.beta1,center=True,scale=True)
			y = tf.nn.leaky_relu(y,alpha = 0.2)
			y = tf.layers.dense(y, 1, activation=None)
			#y = tf.layers.dense(y, 1, activation=tf.nn.sigmoid) 
				
		return y
		
	
	def discriminator_conv(self,y,reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			y = tf.reshape(y, [-1, 1024,3])
			y = tf.contrib.layers.conv1d(y,num_outputs =64,kernel_size=40,stride=1,padding="same",activation_fn=tf.nn.leaky_relu)
			y = tf.contrib.layers.conv1d(y,num_outputs =128,kernel_size=20,stride=2,padding="same",activation_fn=tf.nn.leaky_relu)
			y = tf.contrib.layers.conv1d(y,num_outputs =256,kernel_size=10,stride=2,padding="same",activation_fn=tf.nn.leaky_relu)
			y = tf.contrib.layers.conv1d(y,num_outputs =512,kernel_size=10,stride=1,padding="same",activation_fn=tf.nn.leaky_relu)			
			
			y = tf.layers.dense(y, 128, activation=tf.nn.leaky_relu)				
			y = tf.layers.dense(y, 64, activation=tf.nn.leaky_relu)
			y = tf.layers.dense(y, 1, activation=tf.nn.leaky_relu)				
				
		return y	
			
	def build_network(self):		
		self.input = tf.placeholder(tf.float32, [None,self.pointcloud_dim,3], name="real_pointcloud_data")
		self.z = tf.placeholder(tf.float32,[None,self.z_dim], name ="noice")
		self.Gen = self.generator(self.z)		
		self.Dis_real = self.discriminator_conv(self.input,reuse = False)				
		self.Dis_fake = self.discriminator_conv(self.Gen,reuse = True)		
		
		
		#Tensorboard variables
		
		self.d_sum_real = tf.summary.histogram("d_real", self.Dis_real)
		self.d_sum_fake = tf.summary.histogram("d_fake", self.Dis_fake)
		self.G_sum = tf.summary.histogram("G",self.Gen)
		self.z_sum = tf.summary.histogram("z_input",self.z)
		
		
		#Wassersteinmetrik
		self.d_loss = tf.reduce_mean(self.Dis_fake - self.Dis_real)
		self.g_loss = tf.reduce_mean(-self.Dis_fake)
		
		
		#Vanilla GAN
		#self.d_loss = tf.reduce_mean(-tf.log(self.Dis_real) - tf.log(1. - self.Dis_fake))
		#self.g_loss = tf.reduce_mean(-tf.log(self.Dis_fake))
		
		tf.summary.scalar('self.g_loss', self.g_loss )		
		tf.summary.scalar('self.d_loss', self.d_loss )
		self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
		self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
		self.saver = tf.train.Saver()
		
		#Tensorboard variables
		self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summary_d_loss = tf.summary.scalar("d_loss",self.d_loss)
		


		
	def train(self):
		print("start training")
		self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.d_loss,var_list = self.vars_D)		
		print("d_optim")
		self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.g_loss,var_list = self.vars_G)
		print("g_optim")
		
		with tf.Session() as sess:


			
			#imported_meta = tf.train.import_meta_graph("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4.meta")
			#imported_meta.restore(sess, "C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4")
			
			train_writer = tf.summary.FileWriter("./logs",sess.graph)
			merged = tf.summary.merge_all()
			#test_writer = tf.summary.FileWriter("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/")
			self.leaf_counter = 1
			self.counter = 1		
			sess.run(tf.global_variables_initializer())			
			self.training_data = create_training_data_example()					
			k = (len(self.training_data) // self.batch_size)			
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]
			print("Lengh of the training_data:")
			print(len(self.training_data))
			
				
			
			for e in range(1,self.epoch):
				epoch_loss_d = 0.
				epoch_loss_g = 0.
			
			
				print("data shuffeld")
				np.random.shuffle(self.training_data)		
				for i in range(0,k):
					self.batch_z = np.random.uniform(0, 0.2, [self.batch_size, self.z_dim])				
					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					
					#fake = sess.run([self.Gen] , feed_dict = {self.z: self.batch_z})					
					#fake = sess.run([self.Dis_fake],feed_dict = {self.z: self.batch_z})
					#real = sess.run([self.Dis_real],feed_dict ={self.input: self.batch})					
					#fake = sess.run([self.Dis_fake],feed_dict={self.z: self.batch_z})
					#real = sess.run([self.Dis_real],feed_dict={self.input: self.batch})
				
					_, loss_d_val,loss_d = sess.run([self.d_optim,self.d_loss,self.summary_d_loss],feed_dict={self.input: self.batch,self.z: self.batch_z})
					train_writer.add_summary(loss_d,self.counter)
					_, loss_g_val, loss_g = sess.run([self.g_optim,self.g_loss,self.summary_g_loss],feed_dict={self.z: self.batch_z})
					train_writer.add_summary(loss_g,self.counter)
					_, loss_g_val = sess.run([self.g_optim,self.g_loss],feed_dict={self.z: self.batch_z})					
					self.counter=self.counter + 1					
					epoch_loss_d += loss_d_val
					epoch_loss_g += loss_g_val
					#print(loss_g_val)
					 
				
				epoch_loss_d /= k
				epoch_loss_g /= k
				#print("fake")
				#print(fake)
				#print("real")
				#print(real)
				print("Loss of D: %f" % epoch_loss_d)
				print("Loss of G: %f" % epoch_loss_g)
				print("Epoch%d" %(e))
				
				if e % 100 == 0:
					save_path = self.saver.save(sess,"C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt",global_step=e)
					print("model saved: %s" %save_path)
					self.gen_noise = np.random.uniform(0, 0.2, [1, self.z_dim])
					test_leaf = sess.run([self.Gen], feed_dict={self.z:self.gen_noise})					
					save_leaf(test_leaf,self.leaf_counter,self.leaf_fake)										
					print("created fake_test_leaf")
					real_leaf = self.training_data[random.randrange(0,self.batch_size*k)]
					save_leaf(real_leaf,self.leaf_counter,self.leaf_real)
					print("created real_test_leaf")
					self.leaf_counter= self.leaf_counter + 1
			print("training finished")


            

           
    
        



        
