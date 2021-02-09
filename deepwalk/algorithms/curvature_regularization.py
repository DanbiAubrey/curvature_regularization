import sys
from geopy.distance import geodesic
import numpy as np

import matplotlib.pyplot as plt
import math
from itertools import combinations

#randomwalk based curvature regularization
#generate random walks for entire nodes

#2-dim에만 해당되는 것인지(o) resolved!!!!

#For theorem.1

class abs_curvature_regularization:
    def __init__(self, walks, two_dim_embedding, num_walks, num_nodes, W, dimension):
        self.N = dimension#64(karate.dataset)
        self.X_train = [] 
        self.y_train = [] 
        self.V = num_nodes#34(karate.dataset)
        self.trained_W = W
        self.alpha = 0.001
        self.walks = walks#e.g., [['24', '24', '25', '25', '24', '31', '32', '18', '18', '33'],[...],...]
        self.two_dim_embeddings = two_dim_embedding#[[0.9873423, 0.75242349],[...],...]]
        self.vectorized_walks = self.node_to_vector()#{33: [-0.5739610893522015, -0.12715181451182705], ...0: [0.8261685153358155, 0.13989395141095784]}]
        
    def curvature_regularization(self):

        abs_of_randwalks = self.get_total_abs()
        
        #minimize sum of abs_for_ranwalks

        #satisfy Theorem.1
        sum_of_abs_1 = self.curvature_sum(abs_of_randwalks)

        #while i in sum_all_cases < 90: # minimize til any part of P'_{ij} is less than pi/2(90)


    # -----------plot a polygonal curve of randomwalk-----------#            
    #         x_vectors = np.ndarray(len(cur_walk), dtype=np.complex128)#length : 40

    #         y_vectors = np.ndarray(len(cur_walk), dtype=np.complex128)#length : 40

    #         for i in range(len(cur_walk)):

    #             x_vectors[i] = embedding_of_randwalk[i][0]
    #             y_vectors[i] = embedding_of_randwalk[i][1]


    #         print(x_vectors)
    #         print("\n")
    #         print(y_vectors)


    #         plt.figure(figsize=(10, 10))

    #         plt.plot(x_vectors,y_vectors, marker='o')

    #         plt.title('polygonal curve of random_walk[i]')

    #         plt.xlabel('1_dim')
    #         plt.ylabel('2_dim')

    #         plt.show()

    #         exit()

    
    #convert nodes in randomwalk to 2-dim vector
    def node_to_vector(self):
        node_embeddings = self.two_dim_embeddings
        walks = self.walks
        vectorized_walks = self.walks


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
    def angle(self, vector1, vector2, vector3):
        #print("vector1:{}".format(vector1))
        v0 = np.array(vector2) - np.array(vector1)
        v1 = np.array(vector3) - np.array(vector2)
        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

        return np.degrees(angle)

    def get_total_abs(self):

        node_embeddings = list(self.two_dim_embeddings)
        #ABS curvature(turning angle for each point in randomwalk r)
        total_turning_angles = []#each item : each random_walk / in each item : {key = node_num : val = 2-dim vector}
        
        for i in range(len(self.walks)):#iter 340
            cur_walk = self.vectorized_walks[i]

#             print(cur_walk)#{7: [0.793727189595323, 0.35655434913847506]}

            turning_angles = {}
            if len(cur_walk) > 2:
                keys = cur_walk.keys()
                for q in range(0, len(cur_walk)-2):#range(0,8)
                    node_angle = self.angle(cur_walk[list(keys)[q]], cur_walk[list(keys)[q+1]], cur_walk[list(keys)[q+2]])
                    
                    #print(cur_walk[list(keys)[q]])
                    node_number = list(keys)[q+1]
                    turning_angles[node_number] = node_angle

                total_turning_angles.append(turning_angles)
                
        return total_turning_angles

    ####-------------------------------------------------------------------------------####
    ####                          curvature regularization term                        ####
    ####-------------------------------------------------------------------------------####
    #sum of cosine of curvatures in all cases(any part of P'_{ij})
    def curvature_sum(self, total_turning_angles):
        sum_of_angles = []

        for i in total_turning_angles:
            turning_angles = i
            for j in turning_angles.values():
                cosine_val = math.cos(j)
                sum_of_angles.append(cosine_val)

        total_sum = sum(sum_of_angles)

        return sum_of_angles

    def initialize(self): 
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V)) # weight matrix_1(64, 34)
        
    def feed_forward(self,X): 
        self.h = np.dot(self.W.T,X).reshape(self.N,2) #shape(34,2)
        self.u = np.dot(self.W1.T,self.h)#(34,2) 
        #print(self.u)   
        return self.u
    
#     def backpropagate(self, sum_of_abs): #train by backpropagating
#         #loss function
#         e = 0 - sum_of_abs
#         e = self.y - np.asarray(t).reshape(self.V,2) #(34,2) - (34,2)
#         # e.shape is V x 1 
#         dLdW1 = np.dot(self.h,e.T) 
#         X = np.array(x).reshape(self.V,1) 
#         dLdW = np.dot(X, np.dot(self.W1,e).T) 
#         self.W1 = self.W1 - self.alpha*dLdW1 
#         self.W = self.W - self.alpha*dLdW