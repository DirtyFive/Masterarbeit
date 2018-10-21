from pyntcloud import PyntCloud
import os
import trimesh
import math
import numpy as np
import tensorflow as tf
from random import shuffle
import pandas as pd
from plyfile import PlyData, PlyElement

from random import shuffle



def save_leaf(leaf,counter,leaf_name):
	leaf = np.asarray(leaf)
	leaf = np.reshape(leaf,(1024,3))
	leaf_final = []
	x = 0
	for e in enumerate(leaf):
		leaf_final.append(tuple(leaf[x]))
		x = x +1
	vertex = np.array(leaf_final,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
	el = PlyElement.describe(vertex, 'vertex')
	PlyData([el]).write('%s%d.ply' % (leaf_name,counter))

# produce the Data

def shuffle_trainingdata(training_data):
	ind_list = [i for i in range(training_data)]
	shuffle(ind_list)
	train_new  = training_data[ind_list, :,:,:]
	
    
def rotate(theta):
        R_x = np.array([[1,0,0],
                        [0,math.cos(theta[0]),-math.sin(theta[0])],
                        [0,math.sin(theta[0]),math.cos(theta[0])]
                        ])
        R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],
                        [0,1,0],
                        [-math.sin(theta[1]),0,math.cos(theta[1])]
                        ])
        R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],
                        [math.sin(theta[2]),math.cos(theta[2]),0],
                        [0,0,1]
                        ])
        R = np.dot(R_z, np.dot( R_y, R_x ))

        return R

             

def create_rotator():
        rotater_list =[[30,40,40],
                       [20,89,12],
                       [12,12,12],
                       [90,90,69],
                       [13,12,60],
                       [0,0,0],
                       [20,10,40],
                       [11,12,14],
                       [90,180,290],
                       [180,90,329],
					   [90,90,90],
					   [70,50,60],
					   [60,20,22],
					   [50,180,90],
					   [180,180,180],
					   [200,200,200],
					   [20,20,30],
					   [40,40,40],
					   [20,20,50],
					   [20,30,40]]
        rotater = []
        for i in range(0,len(rotater_list)):
                rotater.append(rotate(rotater_list[i]))
        return rotater
        

def create_training_data():
	training_data = []
	print("Start loading training_data")
	rotater = create_rotator()
	counter=0
	for k in range(0,441):
		cloud = PyntCloud.from_file("B%d.ply" % (k))
		cloud.add_scalar_field("hsv")
		voxelgrid_id = cloud.add_structure("voxelgrid", x_y_z=[128,128,128])
		points = cloud.get_sample("voxelgrid_centroids",voxelgrid_id=voxelgrid_id)
		new_cloud = PyntCloud(points)
		cloud1_test = new_cloud.get_sample(name="points_random",n = 1024)
		new_cloud = PyntCloud(cloud1_test)
		xyz_load = np.asarray(new_cloud.points)
		#print(xyz_load)
		training_data.append([xyz_load])
		#new_cloud.to_file("out_file_%d.ply" % (k))		
        #for r in range(0,len(rotater)):          
            #training_data.append([])
            #for i in range(0,len(xyz_load)):                
                #c = np.dot(xyz_load[i],rotater[r])
                #training_data[counter].append([c[0],c[1],c[2]])            
            #counter = counter + 1
	print(len(training_data))
	print("data loaded")
	print("shuffle training data")
	training_data = np.asarray(training_data)
	print("getting Trainingdata into the right format")
	training_data = training_data.reshape(441,3072)
	print(" trainingdata formated")
	return training_data
	
	
def create_training_data_example():
	training_data = []
	counter = 1
	for file in os.listdir("F:/punktwolkenplot/shape_net_core_uniform_samples_2048/table"):
		if file.endswith(".ply"):
			cloud = PyntCloud.from_file("F:/punktwolkenplot/shape_net_core_uniform_samples_2048/table/%s" % file)        
			#cloud.add_scalar_field("hsv")
			#voxelgrid_id = cloud.add_structure("voxelgrid", x_y_z=[128,128,128])
			#points = cloud.get_sample("voxelgrid_centroids",voxelgrid_id=voxelgrid_id)   
			#new_cloud = PyntCloud(points)
			cloud1_test = cloud.get_sample(name="points_random",n = 1024)
			new_cloud = PyntCloud(cloud1_test)    
			cloud_array = np.asarray(new_cloud.points)   
			#new_cloud.to_file("C:/Users/Andreas/Desktop/jupyter notebook/table/out_file_%d.ply" % (counter))  
			counter = counter + 1
			training_data.append(cloud_array)			
	print(len(training_data))
	print("data loaded")
	print("shuffle training data")
	training_data = np.asarray(training_data)
	print(training_data.shape)
	print("getting Trainingdata into the right format")
	#training_data = training_data.reshape(8509,3072)
	print(training_data.shape)
	print(" trainingdata formated")
	return training_data