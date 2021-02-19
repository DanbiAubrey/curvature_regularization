#!/usr/bin/env python
# coding: utf-8

# ## README
# 
# This is DeepWalk implementation by Danbi with Karate dataset.
# 

# In[1]:


#Import required Libraries
import os
import sys
import random
import argparse
import time
import pandas as pd

import curvature_regularization
import mygraph
from language_model import Skipgram

from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Vocab
from multiprocessing import cpu_count
import networkx as nx
import numpy as np
import scipy.sparse as sp
import cmath

import logging
from gensim.models.callbacks import CallbackAny2Vec



# In[ ]:

def __get_logger():

    __logger = logging.getLogger('logger')

    formatter = logging.Formatter('LOG##LOGSAMPLE##%(levelname)s##%(asctime)s##%(message)s >> @@file::%(filename)s@@line::%(lineno)s')

    stream_handler = logging.StreamHandler()

    stream_handler.setFormatter(formatter)

    __logger.addHandler(stream_handler)

    __logger.setLevel(logging.INFO)

    return __logger

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1
        
#DeepWalk process
def deepwalk_process(args):

  start_time = time.time()#processing time measurement

  logger = __get_logger()
    
  if args.format == "adjacency":
    graph_adjacency, num_nodes, num_edges = text_to_adjacency(args.input)
    G = mygraph.Graph(graph_adjacency, num_nodes, num_edges)#graph object
    
  print("\nNumber of nodes: {}".format(G.num_of_nodes))
  print("\nNumber of edges: {}".format(G.num_of_edges))
    
  num_walks = G.num_of_nodes * args.number_walks
    
  print("\nNumber of walks: {}".format(num_walks))
    
  data_size = num_walks * args.walks_length

  print("\nData size (walks*length): {}".format(data_size))
    
  #Embedding phase  
  print("\nWalking...")
  #shape(340 x 40)
  walks = G.build_deep_walk(num_paths=args.number_walks, path_length=args.walks_length, 
                            alpha=0, rand=random.Random(args.seed))

  print("\nCounting vertex frequency...")
  vertex_counts = count_words(walks)# dictionary

  print("\nTraining...")
  if args.model == 'skipgram':
    #create skipgram model
    language_model = Skipgram(sentences=walks, vocabulary_counts=vertex_counts,size=args.dimension,
                     window=args.window_size, min_count=0, trim_rule=None, workers=cpu_count(), compute_loss=True, callbacks=[callback()])
    
    #save skipgram model
    language_model.save("skipgram_model")
    
    #reload skipgram model
    model = Skipgram.load("skipgram_model")
    
    ####-------------------------------------------------------------------------------####
    ####                              embedding Generation                             ####
    ####-------------------------------------------------------------------------------####
    
    # for t iterations do
    for t in range(args.epoch):
        # while not converged do -> minimize embedding loss term
        model.train(sentences=walks, total_examples=1, epochs=args.epoch, compute_loss=True, callbacks=[callback()])
        model.wv.save_word2vec_format(args.output)

        # load embeddings
        embedding_results = {}
        for i in range(G.num_of_nodes):
            embedding_results[i] = list(model.wv[str(i)])
            #embedding_results.append(model.wv[str(i)])
        
        embedding_dim = len(embedding_results[0])
        
        #64-dim embeddings (shale(34*64))
        original_embedding = []
        
        for i in list(embedding_results.keys()):
            original_embedding.append(embedding_results[i])
            
            
#         print(original_embedding)
#         print(np.array(original_embedding).shape)
#         exit()
            
           
    ####-------------------------------------------------------------------------------####
    ####                                     PCA                                       ####
    ####-------------------------------------------------------------------------------####       
        
        # convert n-dimensional embedding to 2-dim(to satisfy Theorem 1.)
        df = pd.DataFrame(columns = range(0,embedding_dim))

        for i in range(G.num_of_nodes):
            df.loc[i] = embedding_results[float(i)]
    
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

        ####-------------------------------------------------------------------------------####
        ####                          curvature regularization phase                       ####
        ####-------------------------------------------------------------------------------####
        
        # while not converged do -> minimize curvature regularization term
        
        #generate random walks(walk length :5)
        
        walks_2 = G.build_deep_walk_for_abs(num_paths=args.number_walks, path_length=args.walks_length_2, 
                            alpha=0, rand=random.Random(args.seed))
        
        curvature_reg_model = curvature_regularization.abs_curvature_regularization(walks_2, 
                                                                                    neww_X, num_walks,G.num_of_nodes, model.syn1, args.dimension, original_embedding)
        
        # meet the condition of Theorem 1.
        curvature_reg_model.optimization()
        
        #minimize the two terms jointly

    
  else:
    raise Exception('language model is not Skipgram')
    
  total_time = time.time() - start_time

#   print("\nTraining completed")
#   print("\nembeddings have been generated")
#   
  print("\nProcessing time: {:.2f}".format(total_time))


# In[ ]:


def text_to_adjacency(input_graph_file):#change the arg at the end 
    with open(input_graph_file, 'r') as f: 

#         lines = f.readlines()
#         print(lines)
        
        num_lines= sum(1 for line in f)#number of nodes
        #print(num_lines)
        graph = []
            
        for i in range(num_lines):#create 34 * 34 0 entry list
            graph.append([0]*num_lines)
            
    num_edges = 0
    with open(input_graph_file, 'r') as f:      
        line_num = 0
        for line in f.readlines():
            nodes = line.split(" ")
            #print(nodes)
            for j in range(len(nodes)):
                n = int(nodes[j]) - 1
                graph[line_num][n] = 1
                #print(line_num, n)
                num_edges += 1
            line_num += 1
    
    sparse_matrix = sp.csr_matrix(graph)#sparse_matrix
    final_graph = nx.from_scipy_sparse_matrix(sparse_matrix)#create a graph from adjacency matrix
    
    #G = nx.from_scipy_sparse_matrix(sparse_matrix)
    #print("{}".format(sparse_matrix))
    return final_graph, num_lines, num_edges

# In[ ]:


def count_words(walks):# to count how many time the words appear in walks
  c = Counter()

  for words in walks:
    c.update(words)
  return c


# In[34]:


#Main
#argument parser
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--format', default='adjacency')#format of input graph file
  parser.add_argument('--input', nargs='?', required=True, help="input graph file")#input graph file
  parser.add_argument('--number-walks', default=10, type=int)#walk length
  parser.add_argument('--walks-length', default=40, type=int)#window size
  parser.add_argument('--window-size', default=5, type=int, help='Window size')
  parser.add_argument('--dimension', type=int, default=34, help='Embeddings dimension(size)')#34
  parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
  parser.add_argument('--model', default='skipgram', help='language modeling(skipgram)')
  parser.add_argument('--seed', default=0, type=int, help='Random seed for random walk')
  parser.add_argument('--output', required=True, help="output embeddings file")
  parser.add_argument('--epoch', required=True, default=1, type=int, help="traning epoch")# t iteration in Alg.1
  parser.add_argument('--walks-length-2', default=40, type=int)#window size
  #parser.add_argument('--number-random-walks', default=33, type=int)#walk length
  

    
  args = parser.parse_args()

  deepwalk_process(args)

if __name__ == "__main__":
  main()


# In[ ]:




