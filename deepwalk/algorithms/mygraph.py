#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import random
from random import shuffle
import scipy.sparse as sp
import networkx as nx


# In[6]:


#Graph utils
class Graph:
    def __init__(self, graph_adjacency, num_nodes, num_edges):
        self.G = graph_adjacency
        self.num_of_nodes = num_nodes
        self.num_of_edges = num_edges
        self.edges = self.G.edges(data=True)
        self.nodes = self.G.nodes(data=True)

# In[8]:

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        g = self.G
        
        if start:
            path = [start]
        else:
            path = [rand.choice(list(g.nodes(data=False)))]#uniform in regard to Nodes while not uniform with edges
        
        while len(path) < path_length:
            current = path[-1]# current node(end node)
            if len(g[current]) > 0:# if there is neighbor of current node
                if rand.random() >= alpha: # if probability of restart is less than random probability
                    path.append(rand.choice(list(g[current].keys())))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]
        
    # In[ ]:

    #build random walks list(shuffle nodes in beforehand)
    def build_deep_walk(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
        g = self.G
        
        walks = []

        #print(g)
        nodes = list(g.nodes)

        for cnt in range(num_paths):
            rand.shuffle(nodes)#shuffle #to speed up the convergence
            for node in nodes:
                walks.append(self.random_walk(path_length, alpha=alpha, rand=rand, start=node))

        return walks


    # In[9]:







