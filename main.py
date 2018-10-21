import numpy as np
import tensorflow as tf

import os
from absl import app
from absl import flags
from pcgan import PCGAN
from helper import create_training_data
import sys

flags = tf.app.flags


flags.DEFINE_integer("epochs",5000,"epochs per trainingstep")
flags.DEFINE_float("learning_rate",0.0001,"learning rate for the model")
flags.DEFINE_integer("z_dim",126,"dimension of the noise Input")
flags.DEFINE_bool("training",True,"running training of the poincloud gan")
flags.DEFINE_string("checkpoint_dir","C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/","where to save the model")
flags.DEFINE_string("sample_dir","samples","where the samples are stored")
flags.DEFINE_integer("number_point",1024,"number of points in each pointcloud")
flags.DEFINE_integer("pointcloud_dim",1024,"number of input")
flags.DEFINE_integer("iterations",100000,"number of patches")
flags.DEFINE_integer("batch_size",50,"size of the batch")
flags.DEFINE_float("beta1",0.5,"adam beta1")
flags.DEFINE_float("beta2",0.5,"adam beta2")
FLAGS = flags.FLAGS

def _main(argv):
	print("initializing Params")	
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
	if FLAGS.training == True:
		("building the Model")
		pcgan = PCGAN(FLAGS.training,FLAGS.epochs,FLAGS.pointcloud_dim,FLAGS.checkpoint_dir,FLAGS.learning_rate,FLAGS.z_dim,FLAGS.batch_size,FLAGS.beta1,FLAGS.beta2)
		pcgan.train()
	else:
		if not pcgan.load(FLAGS.checkpoint_dir):
			print("first train your model")
			
if __name__ == '__main__':
	print('Starting the Programm....')
	app.run(_main)
