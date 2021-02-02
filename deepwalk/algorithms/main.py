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

import mygraph
from language_model import Skipgram

from collections import Counter
from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab
from multiprocessing import cpu_count
import networkx as nx
import numpy as np
import scipy.sparse as sp


# In[ ]:


#DeepWalk process
def deepwalk_process(args):

  start_time = time.time()#processing time measurement

  if args.format == "adjacency":
    graph_adjacency, num_nodes, num_edges = text_to_adjacency(args.input)
    G = mygraph.Graph(graph_adjacency, num_nodes, num_edges)#graph object
    
  print("\nNumber of nodes: {}".format(G.num_of_nodes))
  print("\nNumber of edges: {}".format(G.num_of_edges))
    
  num_walks = G.num_of_nodes * args.number_walks
    
  print("\nNumber of walks: {}".format(num_walks))
    
  data_size = num_walks * args.walks_length

  print("\nData size (walks*length): {}".format(data_size))
    
  print("\nWalking...")
  walks = G.build_deep_walk(num_paths=args.number_walks, path_length=args.walks_length, 
                            alpha=0, rand=random.Random(args.seed))
  
  print("\nCounting vertex frequency...")
  vertex_counts = count_words(walks)# dictionary

  print("\nTraining...")
  if args.model == 'skipgram':
    language_model = Skipgram(sentences=walks, vocabulary_counts=vertex_counts,size=args.dimension,
                     window=args.window_size, min_count=0, trim_rule=None, workers=cpu_count())
    print(language_model)
  else:
    raise Exception('language model is not Skipgram')
    
  total_time = time.time() - start_time

  print("\nTraining completed")
  print("\nembeddings has been generated")
  language_model.wv.save_word2vec_format(args.output)
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
  parser.add_argument('--dimension', type=int, default=64, help='Embeddings dimension(size)')
  parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
  parser.add_argument('--model', default='skipgram', help='language modeling(skipgram)')
  parser.add_argument('--seed', default=0, type=int, help='Random seed for random walk')
  parser.add_argument('--output', required=True, help="output embeddings file")
  #add argument for "window_size=5"
    
  args = parser.parse_args()

  deepwalk_process(args)

if __name__ == "__main__":
  main()


# In[ ]:




