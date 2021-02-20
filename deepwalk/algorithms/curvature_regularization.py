import sys
from geopy.distance import geodesic
import numpy as np

import matplotlib.pyplot as plt
import math
from itertools import combinations
import pandas as pd

import torch
import torch.nn as nn

#randomwalk based curvature regularization
#generate random walks for entire nodes

#For theorem.1

class abs_curvature_regularization:
    def __init__(self, walks, two_dim_embedding, num_walks, num_nodes, W, dimension, original):
        self.dim = dimension#64(karate.dataset)
        self.N = 34#number of neurons
        self.X_train = [] 
        self.y_train = [] 
        self.V = num_nodes#34(karate.dataset)
        self.original_embeddings = original#shape(34,multi_dim(in here 34))
        self.W = W
        self.alpha = 0.01#learning_rate
        self.walks = walks#e.g., [['24', '24', '25', '25', '24', '31', '32', '18', '18', '33'],[...],...] list 
        self.two_dim_embeddings = two_dim_embedding#[[0.9873423, 0.75242349],[...],...]] list
        self.vectorized_walks = node_to_vector(self.two_dim_embeddings, self.walks)#dictionary
    
    #-------------------------------------------------------------------------------#
    #                       meet the condition of Theorem 1.                        #
    #-------------------------------------------------------------------------------#
    
    #--absolute value of summation of curvatures along any part of polygonal curve is less than pi/2--#
    def optimization(self):     
        #--minimize sum of entire_abs til meet theorem 1.--#
        
        # get turning angle of every nodes that appear in each random walk
        abs_for_nodes = get_abs_for_nodes(self.two_dim_embeddings, self.walks, self.vectorized_walks)
        # return summation of absolute value of curvatures for each random walk, shape(198,)
        entire_abs = total_abs_for_each_polygonal(abs_for_nodes)
        # return cosine summation per each randomwalk, shape(198,)
        cosine_sum = sum_cosine_of_curvature(abs_for_nodes)
        
        each_cosine_sum = each_sum_cosine_of_curvature(self.V, abs_for_nodes)
        
#         print(np.array(self.walks).shape)#shape(340,)
#         print(np.array(abs_for_nodes).shape)#shape(198,)
#         print(np.array(cosine_sum).shape)#shape(198,)
        
        self.initialize()
        multi_dim_embeddings = self.original_embeddings
        two_dim_embeddings = self.two_dim_embeddings
        epoch = 0

        # minimize til any part of P'_{ij} is less than pi/2(90 degree)(less than cosine 0)
        while any(entire_abs[i] >= 90 for i in range(len(entire_abs))):
            print("abs:{}".format(entire_abs[0]))
            
            print("W:{}".format(self.W))
            self.train(multi_dim_embeddings, each_cosine_sum)
            
            output = np.dot(self.W.T, multi_dim_embeddings)#shape(34,34)
            output = np.dot(self.W1.T, output)#shape(34,34)
            output = softmax(output)
            
            print("Output:{}".format(output))
            #print("shape:{}".format(output.shape))
            two_dim_embeddings = multi_dim_to_two_dim(output, self.dim, self.V)

            vectorized_walks = node_to_vector(two_dim_embeddings, self.walks)
            abs_for_nodes = get_abs_for_nodes(two_dim_embeddings, self.walks, vectorized_walks)
            
            entire_abs = total_abs_for_each_polygonal(abs_for_nodes)
            cosine_sum = sum_cosine_of_curvature(abs_for_nodes)
            each_cosine_sum = each_sum_cosine_of_curvature(self.V, abs_for_nodes)
            multi_dim_embeddings = output
            epoch += 1
            print("epoch:{}".format(epoch))

        print("epoch:{}".format(epoch))    
        print(multi_dim_embeddings)
        
    #-------------------------------------------------------------------------------#
    #                         minimize two terms jointly                            #
    #-------------------------------------------------------------------------------#
    
    #-- get a flat embedding manifold --#
    #def curvature_regularization(self):

       
   
    #-------------------------------------------------------------------------------#
    #                         update embeddings(regularization)                     #
    #-------------------------------------------------------------------------------#
    def initialize(self): 
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V)) # weight matrix_1(34,34)
        
    def feed_forward(self,X): 
        #####______TODO______##### : recieve 34x34 embeddings(entire embeddings) instead of 34x1
        self.h = np.dot(self.W.T,X).reshape(self.N,1) #shape(N=34,1)
        self.u = np.dot(self.W1.T,self.h)#(34,1) 
        self.y = softmax(self.u)
        #print(self.u)   
        return self.y
    
    def backpropagate(self, x, node_num, each_cosine_sum): #train by backpropagating
        #loss function
        
        #output of last layer
        
#         rd_u = node_to_vector(self.u, self.walks)
#         abs_u = get_abs_for_nodes(self.u, self.walks, rd_u)
#         cos_abs_u = each_sum_cosine_of_curvature(self.V, abs_u)
        
        #ideal_abs = [1]*self.V #shape(34,1)
        
        # give shape
        #cos_abs_u_1 = np.array(cos_abs_u)[np.newaxis]#shape(1,34)
        #ideal_abs_1 = np.array(ideal_abs)[np.newaxis]#shape(1,34)
        cos_abs_u = each_cosine_sum[node_num]
        #print(cos_abs_u)
        
        term = np.array(cos_abs_u)

        #e = self.y - np.asarray(t).reshape(self.V,2) #(34,2) - (34,2)
        # e.shape is V x 1 
        dLdW1 = np.dot(self.h,term) # (34,1) x (1,34)
        X = x #.reshape(self.V,1)#shape(34,1)
        #X = np.array(x).reshape(self.V,1) 
        dLdW = np.dot(X, np.dot(self.W1,term.T).T) 
        self.W1 = self.W1 - self.alpha*dLdW1 
        self.W = self.W - self.alpha*dLdW
        
    def train(self, original_embeddings, each_cosine_sum): #training process with given epochs
        #for x in range(1,epochs):         
        self.loss = 0 #initialize loss
        
        for i in range(len(original_embeddings)):#shape(34,34)
            self.feed_forward(original_embeddings[i]) 
            self.backpropagate(original_embeddings[i], i, each_cosine_sum) #two_dim_embeddings[i] : shape(1,34)
 
        
#-------------------------------------------------------------------------------#
#                              get_abs functions                                #
#-------------------------------------------------------------------------------#

def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum() 

#convert nodes in randomwalk to 2-dim vector
def node_to_vector(two_dim_embeddings, walks):
    node_embeddings = two_dim_embeddings
    walks = walks
    vectorized_walks = walks


    for i in range(len(walks)):#iter 340(340 number of random_walks: 10 for each 34 nodes)
        #for each random_walk(current random_walk)
        cur_walk = walks[i]#length:10

        #calculate ABS curvature of each node q in i random_walk 
        walk_dict = {}
        for q in cur_walk:
            #get 2-dim embedding of node q
            walk_dict[int(q)] = list(node_embeddings[int(q)])

        vectorized_walks[i]= walk_dict

    return vectorized_walks
#calculate angle between two vectors
def angle(vector1, vector2, vector3):
    v0 = np.array(vector2) - np.array(vector1)
    v1 = np.array(vector3) - np.array(vector2)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

    angle_degree = np.degrees(angle)
    #angle_radian = np.radians(angle)
    return angle_degree

def get_abs_for_nodes(two_dim_embeddings, walks, vectorized_walks):
    node_embeddings = list(two_dim_embeddings)
    #ABS curvature(turning angle for each point in randomwalk r)
    #each item : each random_walk / in each item : {key = node_num : val = 2-dim vector}
    total_turning_angles = []

    for i in range(len(walks)):#iter 340
        cur_walk = vectorized_walks[i]

        #print(cur_walk)#{7: [0.793727189595323, 0.35655434913847506]}

        turning_angles = {}
        if len(cur_walk) > 2:
            keys = cur_walk.keys()
            for q in range(0, len(cur_walk)-2):#range(0,8)
                node_angle = angle(cur_walk[list(keys)[q]], cur_walk[list(keys)[q+1]], cur_walk[list(keys)[q+2]])

                node_number = list(keys)[q+1]
                turning_angles[node_number] = node_angle

            total_turning_angles.append(turning_angles)

    return total_turning_angles

#return summation of cosine value of absolute value of curvatures for each random walk |\sum_{p} K_{p}(P'_{i,j}^{s})|    
def total_abs_for_each_polygonal(entire_abs):
    absolute_sum = []
        #for each set of abs in each random walk
    for i in entire_abs:
        sum_of_abs = 0
        nodes = list(i.keys())
        for key in nodes:
            sum_of_abs += np.absolute(i[key])
        absolute_sum.append(sum_of_abs)
            
    return absolute_sum

#-------------------------------------------------------------------------------#
#                          curvature regularization term                        #
#-------------------------------------------------------------------------------#
    
#sum of cosine of curvatures in all cases(any part of P'_{ij})
def sum_cosine_of_curvature(entire_abs):
    absolute_sum = []
        #for each set of abs in each random walk
    for i in entire_abs:
        sum_of_abs = 0
        nodes = list(i.keys())
        for key in nodes:
            #sum_of_abs += math.cos(np.radians(np.absolute(i[key])))
            sum_of_abs += math.cos(np.absolute(np.radians(i[key])))
        absolute_sum.append(sum_of_abs)
            
    return absolute_sum

#sum of cosine of curvatures in all cases(any part of P'_{ij})
def each_sum_cosine_of_curvature(V, total_turning_angles):
    sum_of_angles = [0] * V

    for i in total_turning_angles:
        turning_angles = i#each dictionary
        for j in turning_angles.keys():
            cosine_val = math.cos(np.absolute(np.radians(turning_angles[j])))
            sum_of_angles[j] += cosine_val

    return sum_of_angles

def multi_dim_to_two_dim(embedding_results, dim, num_nodes):
            # convert n-dimensional embedding to 2-dim(to satisfy Theorem 1.)
    embedding_dim = dim
    df = pd.DataFrame(columns = range(0,embedding_dim))

    for i in range(num_nodes):
        df.loc[i] = embedding_results[i]

    #print(df)

    #Implement PCA to reduce dimensionality of embeddings

    #vector representation(embeddings) list
    X = df.values.tolist()
    #print(X)
    #Computing correlation of matrix
    X_corr=df.corr()

    #Computing eigen values and eigen vectors
    values,vectors=np.linalg.eig(X_corr)

    #Sorting the eigen vectors coresponding to eigen values in descending order
    arg = (-values).argsort()
    values = vectors[arg]
    vectors = vectors[:, arg]

    #Taking first 2 components which explain maximum variance for projecting
    new_vectors=vectors[:,:2]

    #Projecting it onto new dimesion with 2 axis
    neww_X=np.dot(X,new_vectors)
    neww_X = neww_X.real

    return neww_X